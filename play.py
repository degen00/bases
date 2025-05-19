from bases import Bases, train_agents, hyperparameter_tuning

if __name__ == "__main__":
    size = int(input("Enter the size (greater than 2) of the grid: ")) - 1
    mode = input("Options: 'train', 'tune', or 'play': ").strip().lower()

    if mode == 'train':
        episodes = int(input("Enter the no. of training episodes: "))
        train_agents(size, episodes)
        print("Training completed. Policy saved to 'policy.json'.")
    elif mode == 'tune':
        iterations = int(input("Enter the no. of hyperparameter trials: "))
        train_episodes = int(input("Enter no. of train episodes per trial: "))
        test_episodes = int(input("Enter no. of eval episodes per trial: "))
        best_params = hyperparameter_tuning(size, iterations,
                                            train_episodes,
                                            test_episodes)
        print(f"Optimal hyperparameters: {best_params}")
    elif mode == 'play':
        game = Bases(size)
        game.game_loop()
