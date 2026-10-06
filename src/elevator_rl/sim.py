from collections import deque
from dataclasses import dataclass

import numpy as np


class LiftSim:
    def __init__(self, n_floors=10, lift_capacity=8, arrival_probability=0.1, rng=None):
        self._state = None

        self.n_floors = n_floors
        self.floor_probabilities = [arrival_probability for _ in range(n_floors)]
        self.lift_capacity = lift_capacity
        self.rng = rng if rng is not None else np.random.default_rng()

        self.reset()

    def reset(self):
        self._state = LiftState(self.n_floors, self.lift_capacity)

    def state(self):
        return self._state

    def move_up(self):
        if self._state.lift_position < self.n_floors - 1:
            self._state.lift_position += 1

    def move_down(self):
        if self._state.lift_position > 0:
            self._state.lift_position -= 1

    def serve_floor(self):
        n_passengers_served = 0
        n_passengers_who_got_on = 0
        position = self._state.lift_position

        # Remove passengers for this floor
        temp = []
        for passenger in self._state.lift_passengers:
            if passenger.destination == position:
                n_passengers_served += 1
            else:
                temp.append(passenger)
        self._state.lift_passengers = temp

        # Add passengers to lift, first to arrive first, until the lift is full
        waiting = self._state.floor_passengers[position]
        while waiting and len(self._state.lift_passengers) < self.lift_capacity:
            self._state.lift_passengers.append(waiting.popleft())
            n_passengers_who_got_on += 1
        return n_passengers_served, n_passengers_who_got_on

    def sample_passengers(self):
        for floor in range(self.n_floors):
            if self.rng.random() < self.floor_probabilities[floor]:
                # Uniform over every floor except this one
                destination = int(self.rng.integers(self.n_floors - 1))
                if destination >= floor:
                    destination += 1
                self._state.floor_passengers[floor].append(Passenger(floor, destination))

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

    def __init__(self, floors, capacity=8):
        self.capacity = capacity
        self.lift_position = 0
        self.lift_passengers = []
        self.floor_passengers = [deque() for _ in range(floors)]

    def __repr__(self):
        total_string = ""
        lift_string = "[ "
        for passenger in self.lift_passengers:
            lift_string += f"{passenger.destination} "
        for _ in range(self.capacity - len(self.lift_passengers)):
            lift_string += "  "
        lift_string += "] "
        for floor in range(len(self.floor_passengers) - 1, -1, -1):
            string = ""
            if self.lift_position == floor:
                string += lift_string
            else:
                string += " " * len(lift_string)
            string += " | "

            for passenger in reversed(self.floor_passengers[floor]):
                string += f"{passenger.destination} "

            total_string += string + "\n"

        return total_string
