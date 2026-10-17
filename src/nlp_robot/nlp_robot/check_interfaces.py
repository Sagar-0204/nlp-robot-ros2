"""Stage 1 smoke test: prove the custom interfaces were generated and are importable
from Python, and show what a goal / feedback / result looks like.

    ros2 run nlp_robot check_interfaces
"""
from nlp_robot_interfaces.action import ExecuteCommand
from nlp_robot_interfaces.msg import DetectedObject, DetectedObjectArray


def main():
    goal = ExecuteCommand.Goal(command='go to the red box, then come back', dry_run=True)
    fb = ExecuteCommand.Feedback(status='executing', step_index=1, total_steps=2,
                                 current_step='navigate to red_box')
    res = ExecuteCommand.Result(success=True, message='Reached the red box. Back where I started.',
                                plan_json='[{"action": "navigate_to", "target": "red_box"}, '
                                          '{"action": "return_to_start"}]',
                                backend='rules', parse_time_s=0.01)
    obj = DetectedObject(label='red_box', color='red', observations=3, range_source='camera')
    obj.pose.header.frame_id = 'map'
    obj.pose.pose.position.x, obj.pose.pose.position.y = 3.4, 2.4
    arr = DetectedObjectArray(objects=[obj])

    print('Interfaces OK.\n')
    print(f'Goal     : {goal}')
    print(f'Feedback : {fb}')
    print(f'Result   : {res}')
    print(f'Detected : {arr.objects[0].label} at ({obj.pose.pose.position.x}, '
          f'{obj.pose.pose.position.y}), range from {obj.range_source}')


if __name__ == '__main__':
    main()
