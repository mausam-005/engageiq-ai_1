import numpy as np

from src.detection.face_selector import FaceSelector


def create_mock_face(bbox, person_id=0, noise=0.0):
    """Create a mock face dictionary with deterministic landmarks per person_id."""
    np.random.seed(person_id)
    # Base face shape for this person
    landmarks = np.random.rand(468, 3)

    if noise > 0:
        # Add random noise (without seeding) to simulate slight movement
        np.random.seed(None)
        landmarks += np.random.normal(0, noise, size=(468, 3))

    return {"bbox": bbox, "landmarks": landmarks}


def test_single_face_selection():
    """Verify that a single face is always selected as primary."""
    selector = FaceSelector()
    faces = [create_mock_face(bbox=[100, 100, 150, 150])]

    primary_idx = selector.select(faces)
    assert primary_idx == 0
    assert selector.primary_embedding is not None


def test_multi_face_selection_largest_face():
    """Verify that the largest face is selected on the first frame."""
    selector = FaceSelector()
    faces = [
        create_mock_face(bbox=[10, 10, 50, 50]),  # area = 2500
        create_mock_face(bbox=[100, 100, 200, 200]),  # area = 40000 (largest)
        create_mock_face(bbox=[300, 300, 100, 100]),  # area = 10000
    ]

    primary_idx = selector.select(faces)
    assert primary_idx == 1


def test_tracking_persistence():
    """Verify that the same face is tracked across subsequent frames despite size changes."""
    selector = FaceSelector()

    # Frame 1: Face 1 is the largest
    faces_frame1 = [
        create_mock_face(bbox=[10, 10, 50, 50], person_id=1),
        create_mock_face(bbox=[100, 100, 200, 200], person_id=2),  # Largest
    ]
    primary_idx1 = selector.select(faces_frame1)
    assert primary_idx1 == 1

    # Frame 2: Face 0 moves closer and becomes largest, but Face 1 is still present
    # Keep the landmarks very similar to test embedding matching
    faces_frame2 = [
        create_mock_face(
            bbox=[10, 10, 300, 300], person_id=1, noise=0.01
        ),  # Now largest
        create_mock_face(
            bbox=[100, 100, 150, 150], person_id=2, noise=0.01
        ),  # Original primary
    ]

    primary_idx2 = selector.select(faces_frame2)
    # The selector should pick the original primary face based on embedding, not the new largest face
    assert primary_idx2 == 1


def test_face_disappearance_and_reidentification():
    """Verify handling of disappeared faces and timeout expiration."""
    selector = FaceSelector(timeout_seconds=2.0)

    face1 = create_mock_face(bbox=[100, 100, 200, 200], person_id=1)
    face2 = create_mock_face(bbox=[10, 10, 50, 50], person_id=2)

    # Initial selection (t=0)
    assert selector.select([face1], current_time=0.0) == 0

    # Primary face disappears (t=1.0)
    # Only face 2 is present. Since its embedding doesn't match and timeout hasn't expired, returns None
    assert selector.select([face2], current_time=1.0) is None

    # Primary face reappears before timeout (t=1.5)
    assert selector.select([face1, face2], current_time=1.5) == 0

    # Primary face disappears for longer than timeout (t=4.0)
    # Timeout expires, so it should select the largest available face (face2)
    assert selector.select([face2], current_time=4.0) == 0
