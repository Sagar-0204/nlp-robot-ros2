"""nlp_cli: talk to the robot from a terminal. Stage 6.

    ros2 run nlp_robot nlp_cli                                        # interactive
    ros2 run nlp_robot nlp_cli "go to the kitchen, then come back"    # one shot
    ros2 run nlp_robot nlp_cli --dry "turn left and look for the blue box"

While a command runs you see its progress; the robot's own replies arrive on
/nlp_robot/speech. Ctrl+C stops the robot (cancels the command and its Nav2 goal);
Ctrl+C again at the prompt, or "quit", exits. "dry <cmd>" shows the plan without moving.

This is just an action *client* of /execute_command - everything it can do, the stage-5
`ros2 action send_goal` could do too; it only makes it pleasant to use.
"""
import sys
import threading
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String

from nlp_robot_interfaces.action import ExecuteCommand

GREEN, YELLOW, RED, CYAN, DIM, BOLD, RESET = ('\033[32m', '\033[33m', '\033[31m', '\033[36m',
                                              '\033[2m', '\033[1m', '\033[0m')
THINKING_EVERY_S = 5.0


def say(text):
    print(text, flush=True)


class Cli(Node):
    def __init__(self):
        super().__init__('nlp_cli')
        self.client = ActionClient(self, ExecuteCommand, 'execute_command')
        self.create_subscription(String, 'nlp_robot/speech',
                                 lambda m: say(f'{GREEN}robot> {m.data}{RESET}'), 10)
        self.goal_handle = None
        self.done = threading.Event()
        self.done.set()
        self.t0 = 0.0
        self.status = ''
        self.success = False

    def send(self, text, dry=False):
        if not self.client.wait_for_server(timeout_sec=5.0):
            say(f'{RED}/execute_command is not available - is the commander running? '
                f'(terminal 1 should show "nlp_commander: ready", and both terminals need the '
                f'same ROS_DOMAIN_ID){RESET}')
            return False
        self.done.clear()
        self.goal_handle = None
        self.status = 'sent'
        self.t0 = time.monotonic()
        fut = self.client.send_goal_async(ExecuteCommand.Goal(command=text, dry_run=dry),
                                          feedback_callback=self._on_feedback)
        fut.add_done_callback(self._on_goal_response)
        return True

    def cancel(self):
        if self.goal_handle is not None and not self.done.is_set():
            say(f'{YELLOW}stopping...{RESET}')
            self.goal_handle.cancel_goal_async()

    def _on_goal_response(self, fut):
        gh = fut.result()
        if not gh.accepted:
            say(f'{RED}command rejected - another command is still running{RESET}')
            self.done.set()
            return
        self.goal_handle = gh
        gh.get_result_async().add_done_callback(self._on_result)

    def _on_feedback(self, msg):
        f = msg.feedback
        self.status = f.status
        step = f'[{f.step_index}/{f.total_steps}] ' if f.total_steps else ''
        if f.status == 'parsing':
            say(f'{DIM}  ... understanding the command{RESET}')
        elif f.status in ('executing', 'searching'):
            say(f'{CYAN}  > {step}{f.current_step}{RESET}')
        # 'step_done' and 'done' are announced by the robot's own speech / the final result

    def _on_result(self, fut):
        r = fut.result().result
        elapsed = time.monotonic() - self.t0
        self.success = r.success
        if r.success:
            say(f'{GREEN}{BOLD}DONE{RESET}{GREEN} - {r.message}{RESET}')
        elif r.message == 'cancelled':
            say(f'{YELLOW}{BOLD}STOPPED{RESET}{YELLOW} after {elapsed:.1f} s{RESET}')
        else:
            say(f'{RED}{BOLD}NOT DONE{RESET}{RED} - {r.message}{RESET}')
        if r.plan_json and r.plan_json != '[]':
            say(f'{DIM}  plan: {r.plan_json}{RESET}')
        if r.backend:
            say(f'{DIM}  [{r.backend}, planned in {r.parse_time_s:.1f} s]{RESET}')
        self.goal_handle = None
        self.done.set()


def run_and_wait(node, text, dry=False):
    """Send one command and block until it finishes. Ctrl+C cancels it (the robot stops);
    a second Ctrl+C while cancelling gives up waiting."""
    if not node.send(text, dry):
        return False
    cancelled = False
    next_note = time.monotonic() + THINKING_EVERY_S
    while True:
        try:
            if node.done.wait(0.1):
                return node.success
            # The LLM can take a while (CPU, cold model, busy simulation): say so instead
            # of sitting silent, so nobody thinks it hung and presses Ctrl+C.
            if node.status in ('sent', 'parsing') and time.monotonic() > next_note:
                say(f'{DIM}  ... still thinking ({time.monotonic() - node.t0:.0f} s){RESET}')
                next_note += THINKING_EVERY_S
        except KeyboardInterrupt:
            if cancelled:
                raise
            cancelled = True
            node.cancel()


HELP = (f'{BOLD}Type a command for the robot.{RESET} {DIM}Examples: "go to the kitchen, then '
        f'come back", "turn around and back up 30 cm", "find the red box".\n'
        f'Ctrl+C stops the robot, "dry <cmd>" shows the plan only, "quit" exits.{RESET}')


def main(args=None):
    argv = rclpy.utilities.remove_ros_args(sys.argv if args is None else args)[1:]
    # Let Ctrl+C reach us as KeyboardInterrupt instead of rclpy shutting the context down,
    # so the CLI can still send the cancel request to the robot.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = Cli()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    code = 0
    try:
        if argv:
            dry = argv[0] == '--dry'
            ok = run_and_wait(node, ' '.join(argv[1:] if dry else argv), dry=dry)
            code = 0 if ok else 1
        else:
            interactive(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        ex.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(code)


def interactive(node):
    say(HELP)
    while rclpy.ok():
        try:
            line = input(f'\n{BOLD}you> {RESET}').strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            say('')
            break
        if not line:
            continue
        low = line.lower()
        if low in ('quit', 'exit', 'q'):
            break
        if low in ('help', '?'):
            say(HELP)
            continue
        if low in ('stop', 'cancel'):
            say(f'{DIM}nothing is running (press Ctrl+C while a command runs to stop it){RESET}')
            continue
        if low.startswith('dry '):
            run_and_wait(node, line[4:], dry=True)
        else:
            run_and_wait(node, line)


if __name__ == '__main__':
    main()
