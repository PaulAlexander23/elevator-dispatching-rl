from dataclasses import dataclass
from collections import deque
from random import random, choice


class LiftSim:
    def __init__(self):

        self._state = None

        self.max_lift_position = 10
        self.min_lift_position = 0
        self.floor_probabilities = [
            0.1 for n in range(self.max_lift_position - self.min_lift_position)
        ]
        self.lift_capacity = 8

        self.reset()

    def reset(self):
        self._state = LiftState(
            self.max_lift_position - self.min_lift_position, self.lift_capacity
        )
        self._state.lift_position = self.min_lift_position

    def state(self):
        return self._state

    def move_up(self):
        if self._state.lift_position < self.max_lift_position - 1:
            self._state.lift_position += 1

    def move_down(self):
        if self._state.lift_position > self.min_lift_position:
            self._state.lift_position -= 1

    def serve_floor(self):
        n_passengers_served = 0
        n_passengers_who_got_on = 0

        # Remove passengers for this floor
        temp = []
        for passenger in self._state.lift_passengers:
            if passenger.destination == self._state.lift_position:
                n_passengers_served += 1
                # print("passenger got off")
            else:
                temp.append(passenger)
        self._state.lift_passengers = temp

        # Add passengers to lift
        for n in range(self.lift_capacity - len(self._state.lift_passengers)):
            if len(self._state.floor_passengers[self._state.lift_position]) != 0:
                self._state.lift_passengers.append(
                    self._state.floor_passengers[self._state.lift_position].pop()
                )
                n_passengers_who_got_on += 1
                # print("passenger got on")
        return n_passengers_served, n_passengers_who_got_on

    def sample_passengers(self):
        for floor in range(self.max_lift_position - self.min_lift_position):
            if random() < self.floor_probabilities[floor]:
                destination = choice(
                    [
                        n
                        for n in range(self.max_lift_position - self.min_lift_position)
                        if n != floor
                    ]
                )
                self._state.floor_passengers[floor].append(
                    Passenger(floor, destination)
                )

    def render(self):
        print(self.state())


@dataclass
class Passenger:
    origin: int
    destination: int


@dataclass
class LiftState:
    lift_passengers: list[Passenger]
    floor_passengers: list[deque[Passenger]]
    lift_position: int = 0

    def __init__(self, floors, max_persons):
        self.max_persons = max_persons
        self.lift_passengers = []
        self.floor_passengers = []
        for floor in range(floors):
            self.floor_passengers.append(deque())

    def __repr__(self):
        total_string = ""
        lift_string = "[ "
        for passenger in self.lift_passengers:
            lift_string += f"{passenger.destination} "
        for _ in range(self.max_persons - len(self.lift_passengers)):
            lift_string += "  "
        lift_string += "] "
        for floor in range(len(self.floor_passengers) - 1, -1, -1):
            string = ""
            if self.lift_position == floor:
                string += lift_string
            else:
                for n in range(len(lift_string)):
                    string += " "
            string += " | "

            for passenger in reversed(self.floor_passengers[floor]):
                string += f"{passenger.destination} "

            total_string += string + "\n"

        return total_string
