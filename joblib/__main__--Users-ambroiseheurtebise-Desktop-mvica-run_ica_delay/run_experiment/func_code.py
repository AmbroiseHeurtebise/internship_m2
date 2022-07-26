# first line: 119
@mem.cache
def run_experiment(algo_name, n_subjects, p, n, sup_delay, random_state):
    rng = check_random_state(random_state)
    # Initialization
    sigma, delay, mu, var, window_length, peaks_height, sources_height, noise = \
        initialize_model(n_subjects, p, sup_delay, random_state=rng)

    # Create model
    X, A, S, _ = create_model(
        n_subjects, p, n, sigma, delay, mu, var, window_length, peaks_height,
        sources_height, noise, random_state=rng
    )

    # ICA
    if algo_name == 'MVICA':
        _, W_approx, _ = multiviewica(X, random_state=rng)
    elif(algo_name == 'DelayMVICA'):
        _, W_approx, _, _ = delay_multiviewica(X, random_state=rng)
    elif(algo_name == 'GroupICA'):
        _, W_approx, _ = groupica(X, random_state=rng)
    elif(algo_name == 'UniviewICA'):
        W_approx = univiewica(X, random_state=rng)

    # Amari distance
    amari_distances = [amari_distance(w, a) for (w, a) in zip(W_approx, A)]
    mean_amari_distances = np.mean(amari_distances)
    median_amari_distances = np.median(amari_distances)

    # Output
    output = {"Algo": algo_name, "Delay": sup_delay, "random_state": random_state, "Sources": S,
              "Mixing": A, "Unmixing": W_approx, "Amari": amari_distances,
              "Mean_Amari": mean_amari_distances, "Median_Amari": median_amari_distances}
    return output
