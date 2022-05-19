#!/usr/bin/env python3

import sys
import os
import glob
import heapq

from typing import Optional, Callable, Any

from PIL import Image

Size = tuple[int, int]


def area(size: Size) -> int:
    return size[0] * size[1]


def aspect_ratio(size: Size) -> float:
    return size[0] / size[1]


def best_fit(size: Size, aspect_ratio: float) -> Size:
    """Calculate the best fit size for a given aspect ratio."""
    w, h = size

    desired_height = int(w / aspect_ratio)
    if desired_height < h:
        return (w, desired_height)

    desired_width = int(h * aspect_ratio)
    return (desired_width, h)


class Tile(object):
    def __init__(self, filepath, image_size=(0, 0), tile_size=(1, 1)):
        self.image_path = filepath
        self.image_size = image_size

        self.tile_size = tile_size
        self._error = -1.0

    @classmethod
    def from_file(cls, filepath):
        tile = cls(filepath)
        with Image.open(file) as im:
            tile.image_size = (im.width, im.height)
        return tile

    def promote(self, add: Size) -> "Tile":
        """Return a new Tile object with its tile size increased by `add`."""
        new_size = (self.tile_size[0] + add[0], self.tile_size[1] + add[1])
        tile = Tile(self.image_path, image_size=self.image_size, tile_size=new_size)
        return tile

    @property
    def tile_ratio(self) -> float:
        return aspect_ratio(self.tile_size)

    def best_fit(self) -> Size:
        return best_fit(self.image_size, self.tile_ratio)

    def error(self) -> float:
        """Return an error value of the tile aspect ratio relative to the image
        aspect ratio.
        """
        if self._error >= 0:
            return self._error

        resized = best_fit(self.image_size, self.tile_ratio)
        image_area = area(self.image_size)
        resized_area = area(resized)

        rv = self._error = 1.0 - (resized_area / image_area)
        return rv

    def __repr__(self):
        return (
            f"Tile({self.image_path},"
            f" image_size={self.image_size!r},"
            f" tile_size={self.tile_size!r},"
            f" error={self.error()!r})"
        )


class Mosaic(object):
    def __init__(self, size, tiles):
        self.size = size
        self.tiles = tiles

        # Create an w x h grid
        self.grid = [-1] * (size[0] * size[1])

        self._next = 0
        self._empty_tiles = len(self.grid)
        self._placements = None

    def complete(self):
        return self._empty_tiles == 0

    def placements(self):
        result = []

        if not self._placements:
            return result

        for idx, pos in enumerate(self._placements):
            tile = self.tiles[idx]
            result.append((tile, pos))

        return result

    def __lt__(self, other):
        return self._empty_tiles < other._empty_tiles

    def layout(self):
        """Layout out the tiles from left to right, top to bottom."""

        placements = []
        for idx, tile in enumerate(self.tiles):
            if self._next == -1:
                raise ValueError(f"Grid filled prematurely")

            pos = self.find_next_available_slot(tile.tile_size)
            if pos == -1:
                raise ValueError(f"Could not find a position for {tile!r}")

            x, y = self._index_to_cell(pos)

            self.place_at_slot(idx, x, y)
            placements.append((x, y))

            # Update the next available (1, 1) slot
            self._next = self.find_next_available_slot((1, 1))

        # Update the empty tile count
        self._empty_tiles = self._count_empty_tiles()

        # Update placements
        self._placements = placements

    def _count_empty_tiles(self):
        count = 0
        for x in self.grid:
            if x == -1:
                count += 1
        return count

    def _index_to_cell(self, i):
        x = i % self.size[0]
        y = int(i / self.size[0])
        return (x, y)

    def _cell_to_index(self, x, y):
        offset = y * self.size[0]
        return offset + x

    def find_next_available_slot(self, size):
        """Find the next available slot in the grid to fit a tile of size
        `size`.
        """
        grid_w = self.size[0]
        grid_h = self.size[1]

        tile_w, tile_h = size

        for k in range(self._next, len(self.grid)):
            x, y = self._index_to_cell(k)

            x2 = x + tile_w
            if x2 > grid_w:
                # Tile spans outside the right of the grid, ignore
                continue

            y2 = y + tile_h
            if y2 > grid_h:
                # Tile spans beyond the bottom of the grid, ignore
                continue

            # Ensure all slots are empty
            fits = True
            for cx in range(x, x2):
                for cy in range(y, y2):
                    m = self._cell_to_index(cx, cy)
                    if self.grid[m] != -1:
                        fits = False
                        break

            if fits:
                return k

        # Could not find a place to position the tile
        return -1

    def place_at_slot(self, tile_index, x, y):
        tile = self.tiles[tile_index]
        tile_w, tile_h = tile.tile_size

        for cx in range(x, x + tile_w):
            for cy in range(y, y + tile_h):
                m = self._cell_to_index(cx, cy)
                self.grid[m] = tile_index

    def print_grid(self, file=sys.stdout):
        cols = self.size[0]
        rows = self.size[1]

        for y in range(rows):
            for x in range(cols):
                k = self._cell_to_index(x, y)
                value = self.grid[k]
                file.write(f"{value:02d} ")

            file.write("\n")


def sum_of_squares(l):
    return sum(x**2 for x in l)


def mean(l):
    return sum(l) / len(l)


def variance(l):
    avg = mean(l)
    return sum_of_squares(x - avg for x in l) / len(l)


def score_max_area_and_error(m: Mosaic):
    max_error = max([t.error() for t in m.tiles])
    max_area = max([area(t.tile_size) for t in m.tiles])
    return (max_area, max_error)


def score_max_error_and_area(m: Mosaic):
    max_error = max([t.error() for t in m.tiles])
    max_area = max([area(t.tile_size) for t in m.tiles])
    return (max_error, max_area)


class MosaicSolver(object):
    def __init__(
        self,
        grid_size: Size,
        tiles: list[Tile],
        score_fn: Callable[[Mosaic], Any] = score_max_area_and_error,
    ):

        self.grid_size = grid_size
        self.tiles = tiles
        self.score_fn = score_fn

    def solve(self, max_iterations=500) -> Optional[Mosaic]:
        queue = []

        root = Mosaic(self.grid_size, self.tiles)
        root.layout()

        root_score = self.score_fn(root)

        heapq.heappush(queue, (0, root_score, root))

        count = 0
        while queue:
            if count >= max_iterations:
                break
            count += 1

            depth, _, current = heapq.heappop(queue)

            # If solution found, return
            if current.complete():
                return current

            # Promote each tile in the current configuration twice, in each
            # direction.
            tiles = current.tiles

            for idx in range(len(tiles)):
                for add in ((1, 0), (0, 1)):
                    new_tiles = list(tiles)
                    new_tiles[idx] = tiles[idx].promote(add)

                    try:
                        m = Mosaic(self.grid_size, new_tiles)
                        m.layout()
                    except:
                        # Skip if layout fails
                        pass
                    else:
                        score = self.score_fn(m)
                        heapq.heappush(queue, (depth - 1, score, m))

        return None


class MosaicRenderer(object):
    background_color = "#708090"

    def __init__(self, mosaic: Mosaic, size: Size, margin: Size, gutter: int = 0):
        self.m = mosaic
        self.size = size
        self.margin = margin
        self.gutter = gutter

        self.image = None
        self._grid = []
        self._cell_size = (1, 1)

        self._compute_grid()

    def _compute_grid(self):
        cols, rows = self.m.size
        grid = []

        draw_size = (
            self.size[0] - (2 * self.margin[0]),
            self.size[1] - (2 * self.margin[1]),
        )

        # Compute best fit based off the col/rows aspect ratio
        draw_size = best_fit(draw_size, cols / rows)

        # Individual cell sizes
        cell_size = (draw_size[0] // cols, draw_size[1] // rows)

        # Compute top-left starting point
        top_left = (
            (self.size[0] - draw_size[0]) // 2,
            (self.size[1] - draw_size[1]) // 2,
        )

        for y in range(rows):
            row = []
            grid.append(row)

            for x in range(cols):
                offset = (
                    top_left[0] + (cell_size[0] * x),
                    top_left[1] + (cell_size[1] * y),
                )

                row.append(offset)

        self._grid = grid
        self._cell_size = cell_size

    def save(self, path):
        image = Image.new("RGB", self.size, self.background_color)

        for tile, pos in self.m.placements():
            with Image.open(tile.image_path) as tile_image:
                tile_size = (
                    (tile.tile_size[0] * self._cell_size[0]) - self.gutter,
                    (tile.tile_size[1] * self._cell_size[1]) - self.gutter,
                )

                x, y = pos
                offset = self._grid[y][x]
                offset = (
                    offset[0] + (self.gutter // 2),
                    offset[1] + (self.gutter // 2),
                )

                fit = tile.best_fit()
                fit_offset = (
                    (tile.image_size[0] - fit[0]) // 2,
                    (tile.image_size[1] - fit[1]) // 2,
                )

                fit_box = (
                    fit_offset[0],
                    fit_offset[1],
                    fit_offset[0] + fit[0],
                    fit_offset[1] + fit[1],
                )

                resized = tile_image.resize(tile_size, box=fit_box)
                tile_image.close()

                image.paste(resized, offset)
                resized.close()

        image.save(path, quality=95)
        image.close()


# Solve a mosaic configuration for the images in images/
tiles = []

print("Mosaic from images/*.jpg")

try:
    os.mkdir("results")
except:
    pass

for file in glob.glob("images/*.jpg"):
    tiles.append(Tile.from_file(file))

# Sort tiles by filename, ascending.
tiles = sorted(tiles, key=lambda t: os.path.basename(t.image_path))

grids = [
    ((3840, int(3840 * 1.5)), (5, 4)),
    ((3840, int(3840 * 1.5)), (4, 5)),
    ((3840, int(3840 * 1.5)), (3, 6)),
    ((3840, int(3840 * 1.5)), (2, 7)),
]

score_functions = [
    "score_max_area_and_error",
    "score_max_error_and_area",
]

for fnname in score_functions:
    for size, grid in grids:
        fn = globals()[fnname]
        name = f"results/result-{size[0]}x{size[1]}-{grid[0]}x{grid[1]}-{fnname}.jpg"

        solver = MosaicSolver(grid, tiles, score_fn=fn)

        print()
        print(f"-- {fnname} {size[0]}x{size[1]} {grid[0]}x{grid[1]}")

        m = None
        failed = False

        try:
            m = solver.solve()
        except:
            failed = True

        if m is None or failed:
            print("Failed to solve...")
            sys.exit(1)

        m.print_grid()

        render = MosaicRenderer(m, size, (100, 100), 12)
        render.save(name)
