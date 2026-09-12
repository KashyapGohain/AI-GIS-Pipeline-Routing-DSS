import heapq
import math


class AStar:

    def __init__(self, cost_surface):

        self.cost = cost_surface

        self.rows = cost_surface.shape[0]

        self.cols = cost_surface.shape[1]


    # ========================================================
    # HEURISTIC
    # ========================================================

    def heuristic(self, a, b):

        dr = a[0] - b[0]

        dc = a[1] - b[1]

        return math.sqrt(
            dr * dr +
            dc * dc
        )


    # ========================================================
    # NEIGHBOURS
    # ========================================================

    def get_neighbors(self, node):

        r, c = node

        neighbors = [

            (r - 1, c),
            (r + 1, c),

            (r, c - 1),
            (r, c + 1),

            (r - 1, c - 1),
            (r - 1, c + 1),

            (r + 1, c - 1),
            (r + 1, c + 1)
        ]

        valid = []

        for nr, nc in neighbors:

            if (
                0 <= nr < self.rows
                and
                0 <= nc < self.cols
            ):

                valid.append(
                    (nr, nc)
                )

        return valid


    # ========================================================
    # SEARCH
    # ========================================================

    def search(self, start, goal):

        open_set = []

        heapq.heappush(
            open_set,
            (
                0,
                start
            )
        )

        came_from = {}

        g_score = {

            start: 0.0

        }

        closed = set()


        while open_set:

            _, current = heapq.heappop(
                open_set
            )


            if current in closed:

                continue

            closed.add(current)


            # ------------------------------------------------
            # GOAL
            # ------------------------------------------------

            if current == goal:

                path = []

                node = current

                while node in came_from:

                    path.append(node)

                    node = came_from[node]

                path.append(start)

                path.reverse()

                print(
                    "Destination reached!"
                )

                return path


            # ------------------------------------------------
            # NEIGHBOURS
            # ------------------------------------------------

            for neighbor in self.get_neighbors(
                current
            ):

                if neighbor in closed:

                    continue


                nr, nc = neighbor


                cell_cost = float(
                    self.cost[nr, nc]
                )


                # Prevent invalid costs from
                # destroying the search

                if not math.isfinite(
                    cell_cost
                ):

                    continue


                # Distance between cells

                dr = abs(
                    nr - current[0]
                )

                dc = abs(
                    nc - current[1]
                )


                if dr == 1 and dc == 1:

                    movement_cost = math.sqrt(2)

                else:

                    movement_cost = 1.0


                step_cost = (
                    movement_cost *
                    cell_cost
                )


                tentative_g = (
                    g_score[current]
                    +
                    step_cost
                )


                if (
                    neighbor not in g_score
                    or
                    tentative_g <
                    g_score[neighbor]
                ):

                    came_from[
                        neighbor
                    ] = current


                    g_score[
                        neighbor
                    ] = tentative_g


                    f_score = (
                        tentative_g
                        +
                        self.heuristic(
                            neighbor,
                            goal
                        )
                    )


                    heapq.heappush(
                        open_set,
                        (
                            f_score,
                            neighbor
                        )
                    )


        print(
            "WARNING: Destination not reached!"
        )

        return None