"""Coordinated convolutional corrections (DA-CNN research proposal, Oct 2026).

Pilot implementation: WRN-28-4 backbone, two correction sites with four
experts plus an exact zero correction (25 joint actions), alternative-route
loss targets, and additive / joint / bilinear / sequential controllers.
"""

__version__ = "1.0.0"

NUM_CHOICES_DEFAULT = 5  # zero correction + 4 experts per site
