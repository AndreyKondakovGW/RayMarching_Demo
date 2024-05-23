import unittest
from distances import *

class TestDistances(unittest.TestCase):
    def test_clamp(self):
        self.assertEqual(clamp(5, 0, 10), 5)
        self.assertEqual(clamp(-5, 0, 10), 0)
        self.assertEqual(clamp(15, 0, 10), 10)

    def test_sign(self):
        self.assertEqual(sign(5), 1)
        self.assertEqual(sign(-5), -1)
        self.assertEqual(sign(0), 0)

    def test_rotate_point(self):
        # Add test cases for rotate_point function
        pass

    def test_normalize(self):
        self.assertEqual(normalize(3, 4, 12), 13)

    def test_mod(self):
        self.assertEqual(mod(5, 3), 2)
        self.assertEqual(mod(10, 3), 1)

    def test_distance_from_sphere(self):
        self.assertEqual(distance_from_sphere(0, 0, 0, 5, 0, 0, 0), -5)
        self.assertEqual(distance_from_sphere(0, 0, 0, 5, 5, 0, 0), 0)

    def test_distance_from_fuzzy_sphere(self):
        self.assertEqual(distance_from_fuzzy_sphere(0, 0, 0, 5, 0, 0, 0, 1), -5)

    def test_distance_from_planey(self):
        self.assertEqual(distance_from_planey(0, 5, 0, 5), 0)

    def test_distance_from_box(self):
        self.assertEqual(distance_from_box(0, 0, 0, 5, 5, 5, 0, 0, 0, 0), 0)
        self.assertEqual(distance_from_box(0, 6, 0, 5, 5, 5, 0, 0, 0, 0), 1)

    def test_distance_from_frame_box(self):
        self.assertEqual(distance_from_frame_box(0, 0, 0, 5, 5, 5, 0), 0)
        self.assertEqual(distance_from_frame_box(6, 0, 0, 5, 5, 5, 0), 1)

    def test_distance_from_round_box(self):
        self.assertEqual(distance_from_round_box(0, 0, 0, 5, 5, 5, 0), -5)
        self.assertEqual(distance_from_round_box(6, 6, 6, 5, 5, 5, 0), 1)

    def test_distance_from_torus(self):
        self.assertEqual(distance_from_torus(0, 0, 0, 5, 5), -5)
        self.assertEqual(distance_from_torus(0, 5, 0, 5, 5), 0)

    def test_distance_from_cylinder(self):
        self.assertEqual(distance_from_cylinder(0, 0, 0, 5, 10), -5)
        self.assertEqual(distance_from_cylinder(0, 11, 0, 5, 10), 1)

    def test_distance_from_cone(self):
        self.assertEqual(distance_from_cone(0, 0, 0, 0, 0, 10), 0)
        self.assertEqual(distance_from_cone(0, 10, 0, 0, 0, 10), 0)

if __name__ == '__main__':
    unittest.main()
