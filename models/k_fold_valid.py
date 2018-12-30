# This is not used, but it works.

def k_fold_validation (word_input, label_encoded, k=3):
    num_validation_samples = len(word_input) // k

    np.random.shuffle(word_input)

    validation_scores = []
    for fold in range(k):
        validation_data = word_input[num_validation_samples * fold:
                                     num_validation_samples * (fold + 1)]
        validation_label = label_encoded[num_validation_samples * fold:
                                     num_validation_samples * (fold + 1)]
        training_data = np.vstack((
            word_input[:num_validation_samples * fold],
            word_input[num_validation_samples * (fold + 1):]
        ))

        model = get_model()
        model.fit(
            word_input,
            label_encoded,
            epochs=3,
            batch_size=32,
        )
        validation_score = model.evaluate (validation_data, validation_label)[1] # get the accuracy
        validation_scores.append(validation_score)
    validation_score = np.average(validation_scores)