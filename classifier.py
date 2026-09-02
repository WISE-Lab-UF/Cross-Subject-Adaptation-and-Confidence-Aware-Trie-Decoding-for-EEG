import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NUM_CLASSES = 27
ALPHABET_CLASSES = 26
SEED = 42
CALIBRATION_WINDOWS_PER_CLASS = 20

def select_calibration_indices(labels, samples_per_class=CALIBRATION_WINDOWS_PER_CLASS, seed=SEED):
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)

    selected = []

    for class_id in range(NUM_CLASSES):
        class_indices = np.flatnonzero(labels == class_id)

        if len(class_indices) < samples_per_class:
            raise ValueError(f"Class {class_id} has only {len(class_indices)} samples.")

        shuffled = rng.permutation(class_indices)
        selected.extend(shuffled[:samples_per_class].tolist())

    return np.asarray(sorted(selected), dtype=np.int64)

def build_target_classifier():
    # Standardization is fitted only on target calibration embeddings.
    return make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage="auto",
        ),
    )

def fit_target_classifier(calibration_embeddings, calibration_labels):
    classifier = build_target_classifier()

    classifier.fit(np.asarray(calibration_embeddings, dtype=np.float32), np.asarray(calibration_labels, dtype=np.int64))

    return classifier

def probability_matrix(classifier, embeddings, num_classes=NUM_CLASSES):
    raw = classifier.predict_proba(np.asarray(embeddings, dtype=np.float32))

    lda = classifier.named_steps["lineardiscriminantanalysis"]
    classes = np.asarray(lda.classes_, dtype=np.int64)

    full = np.zeros((len(embeddings), num_classes), dtype=np.float64)

    for column, class_id in enumerate(classes):
        full[:, int(class_id)] = raw[:, column]

    full = np.clip(full, 1e-12, None)
    full /= full.sum(axis=1, keepdims=True)

    return full.astype(np.float32)

def alphabetic_probabilities(probabilities):
    # Remove the non-writing class and renormalize a-z.
    probabilities = np.asarray(probabilities, dtype=np.float64)

    alphabetic = np.clip(probabilities[:, :ALPHABET_CLASSES], 1e-12, None)

    alphabetic /= alphabetic.sum(axis=1, keepdims=True)

    return alphabetic.astype(np.float32)


# Use:
#
# calibration_indices = select_calibration_indices(y_target_train)
#
# Z_cal = encoder_512.predict(X_target_train[calibration_indices], verbose=0).astype(np.float32)
#
# Z_test = encoder_512.predict(X_target_test, verbose=0).astype(np.float32)
#
# classifier = fit_target_classifier(Z_cal, y_target_train[calibration_indices])
#
# probabilities_27 = probability_matrix(classifier, Z_test)
#
# probabilities_26 = alphabetic_probabilities(probabilities_27)