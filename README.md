# Elevator dispatching RL 2

I want to apply RL to an lift controller.

## Environment

There are:
- floors - say 10
- lifts - one to start with
- passengers - are at x want to get to y

Actions
- Lift moves up one floor
- Lift moves down one floor
- Lift serves this floor

Transistion

1. Sample new passangers

2. Apply lift action
If the lift is moving up/down, then just update the lift position.
If the lift serves this floor, then:
- Any passengers in the lift that want this floor leave.
- Any passengers waiting enter the lift from first to arrive to last until the lift is full.


Observations
- Lift position
- Those waiting at each floor -> If there are people waiting at that floor
- The number of people in the lift that want each floor -> If people in the lift want that floor
