"""commander: the ExecuteCommand action server. Stage 5.

  text --(CommandParser + LLM)--> validated plan --(this node)--> Nav2 actions

    ros2 action send_goal /execute_command nlp_robot_interfaces/action/ExecuteCommand \
        "{command: 'go to the kitchen, then come back'}" --feedback

Each plan step maps to a Nav2 action:
  navigate_to / go_to_pose / return_to_start -> NavigateToPose
  rotate                                     -> Spin
  move (+/-)                                 -> DriveOnHeading / BackUp
  find (and navigate_to an unseen object)    -> Spin 360, then NavigateToPose over lookout
                                                places, stopping as soon as the detector
                                                reports the object.
Cancelling the ExecuteCommand goal cancels whatever Nav2 action is running.
"""
import json
import math
import threading
import time

import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, PoseStamped
from nav2_msgs.action import BackUp, DriveOnHeading, NavigateToPose, Spin
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration as RclDuration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from nlp_robot_interfaces.action import ExecuteCommand
from nlp_robot_interfaces.msg import DetectedObjectArray

from .command_parser import CommandParser
from .geometry import quat_to_yaw, standoff_goal, yaw_to_quat
from .llm_backends import BackendError, make_backend
from .places import load_places
from .plan_schema import WorldContext, completion_text, describe_step, strip_internal

MOVING = {'navigate_to', 'go_to_pose', 'rotate', 'move', 'find'}


class CommandCancelled(Exception):
    pass


class StepFailed(Exception):
    pass


class Commander(Node):
    def __init__(self):
        super().__init__('nlp_commander')
        p = self.declare_parameter
        p('places_file', '')
        p('llm.backend', 'ollama')
        p('llm.model', '')
        p('llm.base_url', '')
        p('llm.api_key_env', '')
        p('llm.temperature', 0.0)
        p('llm.timeout_s', 180.0)
        p('llm.warmup', True)              # load the model at startup, not on the first command
        p('llm.fallback_backend', 'rules')
        p('global_frame', 'map')
        p('base_frame', 'base_footprint')
        p('object_standoff_m', 0.6)
        p('object_stage_dist_m', 3.0)       # boxes further than this: approach in two stages
        p('object_stage_standoff_m', 1.5)   # first stage stops this far from the box
        p('nav_timeout_s', 180.0)
        p('colors', ['red', 'green', 'blue', 'yellow'])
        p('map_bounds', [-4.0, 4.0, -3.0, 3.0])
        g = lambda n: self.get_parameter(n).value  # noqa: E731

        self.global_frame, self.base_frame = g('global_frame'), g('base_frame')
        self.standoff, self.nav_timeout = g('object_standoff_m'), g('nav_timeout_s')
        self.stage_dist = g('object_stage_dist_m')
        self.stage_standoff = g('object_stage_standoff_m')
        self.colors, self.bounds = list(g('colors')), tuple(g('map_bounds'))
        self.places = self._load_places(g('places_file'))

        llm_kwargs = dict(model=g('llm.model'), base_url=g('llm.base_url'),
                          api_key_env=g('llm.api_key_env'), temperature=g('llm.temperature'),
                          timeout_s=g('llm.timeout_s'))
        primary = make_backend(g('llm.backend'), **llm_kwargs)
        fb = g('llm.fallback_backend')
        fallback = make_backend(fb) if fb and fb != g('llm.backend') else None
        self.parser = CommandParser(primary, self._world_context, fallback=fallback,
                                    logger=lambda m: self.get_logger().warn(m))
        ok, why = primary.check()
        if not ok:
            self.get_logger().error(
                f'LLM backend {primary.describe()} is NOT usable: {why} '
                + ('Commands will use the rule-based fallback until this is fixed.'
                   if fallback else 'Commands will fail until this is fixed.'))
        self.llm_status = 'ok' if ok else f'UNAVAILABLE ({why})'

        self.objects = {}          # label -> (x, y)
        self.anchor = None         # (x, y, yaw) where the last movement command began
        self.busy = False

        self.cb = ReentrantCallbackGroup()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.create_subscription(DetectedObjectArray, 'detected_objects', self._on_objects, 10,
                                 callback_group=self.cb)
        self.speech_pub = self.create_publisher(String, 'nlp_robot/speech', 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.places_pub = self.create_publisher(MarkerArray, 'nlp_robot/places', latched)
        self._publish_place_markers()

        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose', callback_group=self.cb)
        self.spin = ActionClient(self, Spin, 'spin', callback_group=self.cb)
        self.drive = ActionClient(self, DriveOnHeading, 'drive_on_heading',
                                  callback_group=self.cb)
        self.backup = ActionClient(self, BackUp, 'backup', callback_group=self.cb)

        self.server = ActionServer(
            self, ExecuteCommand, 'execute_command',
            execute_callback=self._execute, goal_callback=self._on_goal,
            cancel_callback=lambda _: CancelResponse.ACCEPT, callback_group=self.cb)
        self.get_logger().info(
            f'ready: LLM={primary.describe()} [{self.llm_status}] '
            f'fallback={fallback.describe() if fallback else None}'
            f', {len(self.places)} places. Send goals to /execute_command.')
        if ok and g('llm.warmup'):
            threading.Thread(target=self._warmup, daemon=True).start()

    def _warmup(self):
        """Load the model and process the system prompt once, in the background, so the
        first real command takes ~3 s instead of ~30 s (model load + first prompt)."""
        self.get_logger().info(f'loading {self.parser.backend.describe()}...')
        try:
            secs, where = self.parser.backend.warmup()
            t0 = time.monotonic()
            err = self.parser.prime()
            if err:
                raise BackendError(err)
            self.get_logger().info(
                f'LLM loaded in {secs + time.monotonic() - t0:.1f} s'
                + (f' ({where})' if where else '') + ' - ready for commands')
            if where and 'CPU' in where:
                self.get_logger().warn(f'the model is running {where}: expect several seconds '
                                       f'per command (see `ollama ps`)')
        except BackendError as e:
            self.get_logger().error(f'LLM warm-up failed: {e}. The first command will be slow '
                                    f'or use the fallback.')

    # ------------------------------------------------------------------ setup helpers
    def _load_places(self, path):
        if not path:
            self.get_logger().warn('places_file not set; only boxes and coordinates will work')
            return {}
        return load_places(path)

    def _publish_place_markers(self):
        arr = MarkerArray()
        for i, (name, info) in enumerate(self.places.items()):
            x, y, _ = info['pose']
            for j, mtype in enumerate((Marker.CYLINDER, Marker.TEXT_VIEW_FACING)):
                m = Marker()
                m.header.frame_id = self.global_frame
                m.ns, m.id, m.type = 'places', 2 * i + j, mtype
                m.pose.position.x, m.pose.position.y = float(x), float(y)
                m.pose.orientation.w = 1.0
                if mtype == Marker.CYLINDER:
                    m.scale.x = m.scale.y = 0.25
                    m.scale.z = 0.02
                    m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.6, 1.0, 0.6
                else:
                    m.pose.position.z = 0.35
                    m.scale.z = 0.18
                    m.text = name
                    m.color.r = m.color.g = m.color.b = m.color.a = 1.0
                arr.markers.append(m)
        self.places_pub.publish(arr)

    def _on_objects(self, msg):
        self.objects = {o.label: (o.pose.pose.position.x, o.pose.pose.position.y)
                        for o in msg.objects}

    def _robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(self.global_frame, self.base_frame, Time(),
                                                timeout=RclDuration(seconds=0.5))
        except TransformException:
            return None
        q = t.transform.rotation
        return (t.transform.translation.x, t.transform.translation.y,
                quat_to_yaw(q.x, q.y, q.z, q.w))

    def _world_context(self):
        pose = self._robot_pose()
        return WorldContext(places=self.places, objects=dict(self.objects), colors=self.colors,
                            bounds=self.bounds,
                            robot_pose=None if pose is None else
                            (pose[0], pose[1], math.degrees(pose[2])))

    # ------------------------------------------------------------------ action server
    def _on_goal(self, goal):
        if self.busy:
            self.get_logger().warn('rejecting command: another command is running (cancel it first)')
            return GoalResponse.REJECT
        self.busy = True
        return GoalResponse.ACCEPT

    def _feedback(self, gh, status, idx=0, total=0, text=''):
        fb = ExecuteCommand.Feedback()
        fb.status, fb.step_index, fb.total_steps, fb.current_step = status, idx, total, text
        gh.publish_feedback(fb)

    def _say(self, text):
        self.get_logger().info(f'[robot] {text}')
        self.speech_pub.publish(String(data=text))

    def _execute(self, gh):
        result = ExecuteCommand.Result()
        cmd = gh.request.command.strip()
        try:
            self.get_logger().info(f'command: "{cmd}"')
            self._feedback(gh, 'parsing', text=cmd)
            parsed = self.parser.parse(cmd)
            result.plan_json = json.dumps(strip_internal(parsed.steps))
            result.backend, result.parse_time_s = parsed.backend, float(parsed.latency_s)
            self.get_logger().info(
                f'plan via {parsed.backend} in {parsed.latency_s:.2f}s '
                f'({parsed.attempts} attempt(s)): {result.plan_json}')
            if parsed.fallback_reason:
                self._say(f'(LLM unavailable: {parsed.fallback_reason}. '
                          f'Using the simple rule parser instead.)')
            if not parsed.ok:
                result.success = False
                extra = [e for e in parsed.errors if e not in parsed.reply]
                result.message = parsed.reply + (f' [{"; ".join(extra)}]' if extra else '')
                self._say(parsed.reply)
                (gh.abort if parsed.errors else gh.succeed)()
                return result
            self._say(parsed.reply)
            if gh.request.dry_run:
                result.success, result.message = True, f'(dry run) {parsed.reply}'
                gh.succeed()
                return result

            steps = parsed.steps
            if any(s['action'] in MOVING for s in steps):
                pose = self._robot_pose()
                if pose is not None:
                    self.anchor = pose
            t0 = time.monotonic()
            done = []
            for i, step in enumerate(steps):
                if gh.is_cancel_requested:
                    raise CommandCancelled()
                desc = describe_step(step)
                self.get_logger().info(f'step {i + 1}/{len(steps)}: {desc}')
                self._feedback(gh, 'executing', i + 1, len(steps), desc)
                try:
                    outcome = self._run_step(step, gh, i + 1, len(steps))
                except StepFailed as e:
                    raise StepFailed(f'step {i + 1}/{len(steps)} ({desc}) failed: {e}') from e
                # A step can report what it found ("Found the green box at (3.5, -2.5).");
                # otherwise use the generic completion text ("Reached the kitchen.").
                msg = outcome or completion_text(step)
                if msg:
                    done.append(msg)
                    self._say(msg)
                self._feedback(gh, 'step_done', i + 1, len(steps), msg or desc)
            elapsed = time.monotonic() - t0
            summary = ' '.join(done) if done else parsed.reply
            result.success = True
            result.message = f'{summary} ({len(steps)} step{"s" if len(steps) > 1 else ""}, ' \
                             f'{elapsed:.1f} s)'
            self.get_logger().info(f'command finished: {result.message}')
            self._feedback(gh, 'done', len(steps), len(steps), result.message)
            gh.succeed()
        except CommandCancelled:
            result.success, result.message = False, 'cancelled'
            self._say('Stopped.')
            gh.canceled()
        except (StepFailed, BackendError) as e:
            result.success, result.message = False, str(e)
            self._say(f'I could not finish: {e}')
            gh.abort()
        except Exception as e:  # never leave a goal hanging
            self.get_logger().error(f'unexpected error: {e!r}')
            result.success, result.message = False, f'internal error: {e}'
            gh.abort()
        finally:
            self.busy = False
        return result

    # ------------------------------------------------------------------ steps
    def _run_step(self, step, gh, idx, total):
        """Execute one plan step. May return a sentence describing the outcome."""
        a = step['action']
        if a == 'navigate_to' and step['target_kind'] == 'place':
            x, y, yaw = self.places[step['target']]['pose']
            self._navigate(gh, x, y, math.radians(yaw))
        elif a == 'navigate_to':
            self._go_to_object(gh, step['target'], idx, total)
        elif a == 'find':
            if not self._search(gh, step['target'], idx, total):
                raise StepFailed(f'searched everywhere but did not find the {step["target"]}')
            x, y = self.objects[step['target']]
            return f'Found the {step["target"].replace("_", " ")} at ({x:.1f}, {y:.1f}).'
        elif a == 'go_to_pose':
            pose = self._robot_pose()
            yaw = (math.radians(step['yaw_deg']) if step.get('yaw_deg') is not None
                   else (math.atan2(step['y'] - pose[1], step['x'] - pose[0]) if pose else 0.0))
            self._navigate(gh, step['x'], step['y'], yaw)
        elif a == 'return_to_start':
            if self.anchor is None:
                raise StepFailed('I have no start position to return to yet')
            self._navigate(gh, *self.anchor)
        elif a == 'rotate':
            self._spin(gh, math.radians(step['angle_deg']))
        elif a == 'move':
            self._move(gh, step['distance_m'])
        elif a == 'wait':
            end = time.monotonic() + step['seconds']
            while time.monotonic() < end:
                if gh.is_cancel_requested:
                    raise CommandCancelled()
                time.sleep(0.05)
        elif a == 'say':
            self._say(step['text'])

    def _go_to_object(self, gh, label, idx, total):
        if label not in self.objects:
            self._say(f'I have not seen the {label.replace("_", " ")} yet - searching.')
            if not self._search(gh, label, idx, total):
                raise StepFailed(f'could not find the {label.replace("_", " ")}')
        for attempt in range(3):
            ox, oy = self.objects[label]
            pose = self._robot_pose()
            if pose is None:
                raise StepFailed('robot pose unavailable (is AMCL localised?)')
            if attempt == 0 and math.dist(pose[:2], (ox, oy)) > self.stage_dist:
                # Far boxes are usually camera-only estimates (beyond LiDAR reach). Get within
                # LiDAR range first, then do the precise approach from the refined estimate.
                gx, gy, gyaw = standoff_goal(pose[:2], (ox, oy), self.stage_standoff)
                self.get_logger().info(f'{label} is far away; approaching in two stages')
                self._navigate(gh, gx, gy, gyaw, fail_ok=True)
                continue
            gx, gy, gyaw = standoff_goal(pose[:2], (ox, oy), self.standoff)
            self._navigate(gh, gx, gy, gyaw)
            if attempt == 2:
                return
            nx, ny = self.objects.get(label, (ox, oy))
            if math.hypot(nx - ox, ny - oy) < 0.35:   # estimate stable -> done
                return
            self.get_logger().info(f'{label} estimate moved while approaching; re-planning')

    def _search(self, gh, label, idx, total):
        found = lambda: label in self.objects  # noqa: E731
        if found():
            return True
        self._feedback(gh, 'searching', idx, total, f'spinning to look for {label}')
        here = self._robot_pose()
        self._spin(gh, 2 * math.pi - 0.01, stop_when=found, fail_ok=True)
        if found():
            return True
        # Places already scanned with a full spin; a lookout this close adds nothing
        # (in testing, "find the green box" from home re-spun at home: ~15 s wasted).
        scanned = [here[:2]] if here else []
        remaining = [n for n, i in self.places.items() if i.get('lookout')]
        while remaining:
            pose = self._robot_pose() or (0.0, 0.0, 0.0)
            remaining = [n for n in remaining
                         if all(math.dist(p, self.places[n]['pose'][:2]) > 1.0 for p in scanned)]
            if not remaining:
                break
            remaining.sort(key=lambda n: math.dist(pose[:2], self.places[n]['pose'][:2]))
            name = remaining.pop(0)
            scanned.append(tuple(self.places[name]['pose'][:2]))
            self._feedback(gh, 'searching', idx, total, f'looking for {label} at {name}')
            x, y, yaw = self.places[name]['pose']
            self._navigate(gh, x, y, math.radians(yaw), stop_when=found, fail_ok=True)
            if found():
                return True
            self._spin(gh, 2 * math.pi - 0.01, stop_when=found, fail_ok=True)
            if found():
                return True
        return False

    # ------------------------------------------------------------------ Nav2 wrappers
    def _navigate(self, gh, x, y, yaw, stop_when=None, fail_ok=False):
        goal = NavigateToPose.Goal()
        ps = PoseStamped()
        ps.header.frame_id = self.global_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x, ps.pose.position.y = float(x), float(y)
        qx, qy, qz, qw = yaw_to_quat(float(yaw))
        ps.pose.orientation.z, ps.pose.orientation.w = qz, qw
        goal.pose = ps
        return self._run_action(gh, self.nav, goal, self.nav_timeout,
                                f'navigate to ({x:.2f}, {y:.2f})', stop_when, fail_ok)

    def _spin(self, gh, angle, stop_when=None, fail_ok=False):
        goal = Spin.Goal()
        goal.target_yaw = float(angle)
        goal.time_allowance = Duration(sec=int(10 + abs(angle) * 6))
        return self._run_action(gh, self.spin, goal, goal.time_allowance.sec + 5,
                                f'spin {math.degrees(angle):.0f} deg', stop_when, fail_ok)

    def _move(self, gh, distance):
        fwd = distance > 0
        goal = (DriveOnHeading if fwd else BackUp).Goal()
        goal.target = Point(x=abs(float(distance)))
        goal.speed = 0.15
        goal.time_allowance = Duration(sec=int(10 + abs(distance) / 0.15 * 2))
        return self._run_action(gh, self.drive if fwd else self.backup, goal,
                                goal.time_allowance.sec + 5, f'move {distance:+.2f} m')

    def _wait_future(self, fut, timeout):
        end = time.monotonic() + timeout
        while not fut.done():
            if time.monotonic() > end:
                return False
            time.sleep(0.02)
        return True

    def _cancel_nav(self, nav_gh):
        cancel = nav_gh.cancel_goal_async()
        self._wait_future(cancel, 3.0)

    def _run_action(self, gh, client, goal, timeout, what, stop_when=None, fail_ok=False):
        if not client.wait_for_server(timeout_sec=5.0):
            name = getattr(client, '_action_name', 'Nav2')
            raise StepFailed(f'action server "{name}" is not available - is Nav2 running?')
        send = client.send_goal_async(goal)
        # Wait for Nav2 to accept the goal even if the user cancels meanwhile: if we stopped
        # waiting, a goal accepted a moment later would never be cancelled and the robot
        # would keep driving after saying "Stopped".
        if not self._wait_future(send, 10.0) or not send.result().accepted:
            if gh.is_cancel_requested:
                raise CommandCancelled()
            raise StepFailed(f'Nav2 rejected: {what}')
        nav_gh = send.result()
        res = nav_gh.get_result_async()
        end = time.monotonic() + timeout
        while not res.done():
            stop_early = stop_when is not None and stop_when()
            if gh.is_cancel_requested or stop_early or time.monotonic() > end:
                self._cancel_nav(nav_gh)
                if gh.is_cancel_requested:
                    raise CommandCancelled()
                if stop_early:
                    return True
                if fail_ok:
                    return False
                raise StepFailed(f'timed out: {what}')
            time.sleep(0.05)
        ok = res.result().status == GoalStatus.STATUS_SUCCEEDED
        if not ok and not fail_ok:
            err = getattr(res.result().result, 'error_msg', '') or ''
            raise StepFailed(f'Nav2 failed to {what} {err}'.strip())
        return ok


def main(args=None):
    rclpy.init(args=args)
    node = Commander()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
