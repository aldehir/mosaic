#!/usr/bin/env python3

import sys
import os
import glob
import pprint
import heapq

from typing import Optional

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


def tile_score_average(tiles: list[Tile]):
    """Return back the average error for all tiles."""
    error_sum = sum(t.error() for t in tiles)
    return error_sum / len(tiles)


def tile_score_sum_of_squares(tiles: list[Tile]):
    """Return back the square of sums of tile errors."""
    return sum(int(t.error() * 100) ** 2 for t in tiles)


class Mosaic(object):
    def __init__(self, size, tiles):
        self.size = size
        self.tiles = tiles

        # Create an w x h grid
        self.grid = [-1] * (size[0] * size[1])

        self._next = 0
        self._grid_score = len(self.grid)
        self._tile_score = self._compute_tile_score()
        self._placements = None

    def complete(self):
        return self._grid_score == 0

    def placements(self):
        result = []

        if not self._placements:
            return result

        for idx, pos in enumerate(self._placements):
            tile = self.tiles[idx]
            result.append((tile, pos))

        return result

    def score(self):
        """Return a 2-tuple representing the score of the Mosaic
        configuration."""
        # return (self._grid_score, self._tile_score)
        return (self._tile_score, self._grid_score)

    def __lt__(self, other):
        return self.score() < other.score()

    def _compute_tile_score(self):
        """Return the average error of the tile."""
        return tile_score_sum_of_squares(self.tiles)

    def layout(self):
        """Layout out the tiles from left to right, top to bottom."""

        placements = []
        for idx, tile in enumerate(self.tiles):
            pos = self.find_next_available_slot(tile.tile_size)
            x, y = self._index_to_cell(pos)
            if x == -1:
                raise ValueError(f"Could not find a position for {tile!r}")

            self.place_at_slot(idx, x, y)
            placements.append((x, y))

            # Update the next available (1, 1) slot
            self._next = self.find_next_available_slot((1, 1))

        # Update the grid score
        self._grid_score = len(self.grid)
        for x in self.grid:
            if x != -1:
                self._grid_score -= 1

        # Update placements
        self._placements = placements

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

    def place_at_slot(self, tile_index, col, row):
        tile = self.tiles[tile_index]
        tile_w, tile_h = tile.tile_size

        for j in range(col, col + tile_w):
            for i in range(row, row + tile_h):
                m = self._cell_to_index(j, i)
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


class MosaicSolver(object):
    def __init__(self, grid_size: Size, tiles: list[Tile]):
        self.grid_size = grid_size
        self.tiles = tiles

    def solve(self) -> Optional[Mosaic]:
        queue = []

        root = Mosaic(self.grid_size, self.tiles)
        heapq.heappush(queue, root)

        while queue:
            current = heapq.heappop(queue)
            if current.complete():
                return current

            # Loop through each tile in the current mosaic configuration
            tiles = current.tiles

            for idx in range(len(tiles)):
                for add in ((1, 0), (0, 1)):
                    # Create a copy of the tiles list, but promote tile at
                    # `idx`
                    new_tiles = list(tiles)
                    new_tiles[idx] = tiles[idx].promote(add)

                    # Generate a new mosaic configuration
                    m = Mosaic(self.grid_size, new_tiles)

                    try:
                        m.layout()
                    except:
                        pass
                        # print(f"Failed to layout after promoting tile at {idx} by {add!r}")
                    else:
                        heapq.heappush(queue, m)

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
                offset = (offset[0] + (self.gutter // 2), offset[1] + (self.gutter //2))

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

        image.save(path)
        image.close()


print()
print("-- Mosaic with only 1x1 tiles -- ")
mosaic = Mosaic(
    (5, 3),
    [
        Tile("image-00", image_size=(100, 100), tile_size=(1, 1)),
        Tile("image-01", image_size=(100, 100), tile_size=(1, 1)),
        Tile("image-02", image_size=(100, 100), tile_size=(1, 1)),
        Tile("image-03", image_size=(100, 100), tile_size=(1, 1)),
        Tile("image-04", image_size=(100, 100), tile_size=(1, 1)),
        Tile("image-05", image_size=(100, 100), tile_size=(1, 1)),
        Tile("image-06", image_size=(100, 100), tile_size=(1, 1)),
        Tile("image-07", image_size=(100, 100), tile_size=(1, 1)),
        Tile("image-08", image_size=(100, 100), tile_size=(1, 1)),
        Tile("image-09", image_size=(100, 100), tile_size=(1, 1)),
    ],
)
mosaic.layout()
mosaic.print_grid()

print()
print("-- Mosaic layed out perfectly -- ")
mosaic2 = Mosaic(
    (5, 3),
    [
        Tile("image-00", image_size=(100, 100), tile_size=(2, 2)),
        Tile("image-01", image_size=(100, 100), tile_size=(1, 2)),
        Tile("image-02", image_size=(100, 100), tile_size=(1, 1)),
        Tile("image-03", image_size=(100, 100), tile_size=(1, 2)),
        Tile("image-04", image_size=(100, 100), tile_size=(1, 1)),
        Tile("image-05", image_size=(100, 100), tile_size=(2, 1)),
        Tile("image-06", image_size=(100, 100), tile_size=(2, 1)),
        Tile("image-07", image_size=(100, 100), tile_size=(1, 1)),
    ],
)
mosaic2.layout()
mosaic2.print_grid()

# Solve a mosaic configuration for the images in images/
tiles = []
print("reading in image metadata...")
for file in glob.glob("images/*.jpg"):
    tiles.append(Tile.from_file(file))

# Sort tiles by filename, ascending.
tiles = sorted(tiles, key=lambda t: os.path.basename(t.image_path))

pprint.pprint(tiles)

solver = MosaicSolver((5, 3), tiles)
m = solver.solve()
if m is not None:
    m.print_grid()
else:
    print("Failed to solve...")
    sys.exit(1)

render = MosaicRenderer(m, (1920, 1080), (100, 100), 12)
render.save("test.jpg")
