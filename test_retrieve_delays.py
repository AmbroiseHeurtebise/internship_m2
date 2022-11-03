from delay_multiviewica import generate_data, delay_multiviewica


if __name__ == '__main__':
    # Parameters
    m = 6
    p = 5
    n = 400
    nb_intervals = 5
    nb_freqs = 20
    delay_max = 50
    noise = 0.5
    random_state = 12

    # Generate data
    X_list, _, true_tau_list, _, _ = generate_data(
        m, p, n, nb_intervals=nb_intervals, nb_freqs=nb_freqs,
        treshold=3, delay=delay_max, noise=noise, random_state=random_state)

    # Estimate delays
    _, _, _, tau_list = delay_multiviewica(
        X_list, random_state=random_state)

    # Print
    print(true_tau_list - true_tau_list[0])
    print((n - tau_list) % n)
