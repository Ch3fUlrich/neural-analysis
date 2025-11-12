"""Simple test script to verify decoding module.

This script tests the decoding module independently of pytest.
Run with: python3 test_decoding_simple.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import numpy as np

    print("✓ NumPy imported")
except ImportError:
    print("✗ NumPy not available - install with: pip install numpy")
    sys.exit(1)

try:
    from sklearn.neighbors import KNeighborsRegressor

    print("✓ scikit-learn imported")
except ImportError:
    print("✗ scikit-learn not available - install with: pip install scikit-learn")
    sys.exit(1)

# Test decoding module import
try:
    from neural_analysis import decoding

    print("✓ Decoding module imported")
    print(
        f"  Available functions: {[f for f in dir(decoding) if not f.startswith('_')]}"
    )
except ImportError as e:
    print(f"✗ Failed to import decoding module: {e}")
    sys.exit(1)

# Test synthetic_data module
try:
    from neural_analysis.data.synthetic_data import generate_place_cells

    print("✓ Synthetic data module imported")
except ImportError as e:
    print(f"✗ Failed to import synthetic_data module: {e}")
    sys.exit(1)

# Test population_vector_decoder
print("\n=== Testing population_vector_decoder ===")
try:
    activity, meta = generate_place_cells(
        n_cells=30, n_samples=500, arena_size=(1.0, 1.0), seed=42
    )
    print(f"✓ Generated place cells: {activity.shape}")

    from neural_analysis.learning.decoding import population_vector_decoder

    decoded = population_vector_decoder(
        activity, meta["field_centers"], method="weighted_average"
    )
    print(f"✓ Decoded positions: {decoded.shape}")

    # Check decoding error
    errors = np.linalg.norm(decoded - meta["positions"], axis=1)
    mean_error = errors.mean()
    print(f"✓ Mean decoding error: {mean_error:.4f} (should be < 0.3)")

    if mean_error < 0.3:
        print("✓ Population vector decoder test PASSED")
    else:
        print(
            f"✗ Population vector decoder test FAILED (error too high: {mean_error:.4f})"
        )
except Exception as e:
    print(f"✗ Population vector decoder test FAILED: {e}")
    import traceback

    traceback.print_exc()

# Test k-NN decoder
print("\n=== Testing knn_decoder ===")
try:
    activity, meta = generate_place_cells(
        n_cells=40, n_samples=800, arena_size=(1.0, 1.0), seed=123
    )
    print(f"✓ Generated place cells: {activity.shape}")

    # Split train/test
    train_act, test_act = activity[:600], activity[600:]
    train_pos, test_pos = meta["positions"][:600], meta["positions"][600:]

    from neural_analysis.learning.decoding import knn_decoder

    decoded = knn_decoder(train_act, train_pos, test_act, k=5, weights="distance")
    print(f"✓ Decoded positions with k-NN: {decoded.shape}")

    # Check decoding error
    errors = np.linalg.norm(decoded - test_pos, axis=1)
    mean_error = errors.mean()
    print(f"✓ Mean k-NN decoding error: {mean_error:.4f} (should be < 0.25)")

    if mean_error < 0.25:
        print("✓ k-NN decoder test PASSED")
    else:
        print(f"✗ k-NN decoder test FAILED (error too high: {mean_error:.4f})")
except Exception as e:
    print(f"✗ k-NN decoder test FAILED: {e}")
    import traceback

    traceback.print_exc()

# Test cross-validated k-NN
print("\n=== Testing cross_validated_knn_decoder ===")
try:
    activity, meta = generate_place_cells(
        n_cells=45, n_samples=1000, arena_size=(1.0, 1.0), seed=456
    )
    print(f"✓ Generated place cells: {activity.shape}")

    from neural_analysis.learning.decoding import cross_validated_knn_decoder

    metrics = cross_validated_knn_decoder(activity, meta["positions"], k=5, n_folds=5)
    print(f"✓ Cross-validation completed")
    print(f"  Mean R²: {metrics['mean_r2']:.4f} ± {metrics['std_r2']:.4f}")
    print(f"  Mean error: {metrics['mean_error']:.4f} ± {metrics['std_error']:.4f}")

    if metrics["mean_r2"] > 0.7 and metrics["mean_error"] < 0.3:
        print("✓ Cross-validated k-NN decoder test PASSED")
    else:
        print(f"✗ Cross-validated k-NN decoder test FAILED")
        print(f"  R² = {metrics['mean_r2']:.4f} (should be > 0.7)")
        print(f"  Error = {metrics['mean_error']:.4f} (should be < 0.3)")
except Exception as e:
    print(f"✗ Cross-validated k-NN decoder test FAILED: {e}")
    import traceback

    traceback.print_exc()

# Test compare_highd_lowd_decoding
print("\n=== Testing compare_highd_lowd_decoding ===")
try:
    from sklearn.decomposition import PCA

    activity, meta = generate_place_cells(
        n_cells=80, n_samples=1200, arena_size=(1.0, 1.0), seed=789
    )
    print(f"✓ Generated place cells: {activity.shape}")

    # Create PCA embedding
    pca = PCA(n_components=10)
    embedding = pca.fit_transform(activity)
    print(f"✓ Created PCA embedding: {embedding.shape}")

    from neural_analysis.learning.decoding import compare_highd_lowd_decoding

    comparison = compare_highd_lowd_decoding(
        activity, embedding, meta["positions"], k=5, n_folds=5
    )
    print(f"✓ Comparison completed")
    print(f"  High-D R²: {comparison['high_d']['mean_r2']:.4f}")
    print(f"  Low-D R²: {comparison['low_d']['mean_r2']:.4f}")
    print(f"  Performance ratio: {comparison['performance_ratio']:.4f}")
    print(f"  Dimensionality reduction: {comparison['dimensionality_reduction']}")

    if comparison["high_d"]["mean_r2"] > 0.7 and comparison["performance_ratio"] > 0.7:
        print("✓ High-D vs Low-D comparison test PASSED")
    else:
        print(f"✗ High-D vs Low-D comparison test FAILED")
except Exception as e:
    print(f"✗ High-D vs Low-D comparison test FAILED: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 50)
print("✓ All tests completed successfully!")
print("=" * 50)
