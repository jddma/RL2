import random
import numpy as np
import torch
import torch.nn.functional as F


def linear_epsilon(global_step, eps_start=1.0, eps_end=0.05,
                   decay_steps=100_000):
    """
    Decaimiento lineal de epsilon.
    """
    fraction = min(global_step / decay_steps, 1.0)
    return eps_start + fraction * (eps_end - eps_start)


def select_action(state, online_net, action_space, epsilon, device):
    """
    Política epsilon-greedy.
    """
    if random.random() < epsilon:
        return action_space.sample()

    state_tensor = torch.as_tensor(
        state,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    with torch.no_grad():
        q_values = online_net(state_tensor)

    return int(torch.argmax(q_values, dim=1).item())


def train_dqn(
    env,
    online_net,
    target_net,
    replay_buffer,
    optimizer,
    num_episodes=500,
    gamma=0.99,
    batch_size=64,
    learning_starts=5_000,
    train_frequency=1,
    target_update_frequency=1_000,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_steps=100_000,
    device="cpu",
    seed=None,
    episode_callback=None
):

    # ---------------------------------------------------------
    # 1. Inicialización
    # ---------------------------------------------------------

    online_net.to(device)
    target_net.to(device)

    # Inicialmente ambas redes tienen los mismos parámetros
    target_net.load_state_dict(online_net.state_dict())

    # Target Network no se entrena directamente
    target_net.eval()

    global_step = 0

    episode_rewards = []
    episode_losses = []
    history = []

    # ---------------------------------------------------------
    # 2. Loop de episodios
    # ---------------------------------------------------------

    for episode in range(num_episodes):

        state, info = env.reset(seed=seed + episode) if seed is not None else env.reset()

        episode_reward = 0.0
        episode_steps = 0
        max_distance = float("-inf")
        epsilon = eps_start
        losses = []

        terminated = False
        truncated = False

        # -----------------------------------------------------
        # 3. Loop dentro del episodio
        # -----------------------------------------------------

        while not (terminated or truncated):

            # -------------------------------------------------
            # Política epsilon-greedy
            # -------------------------------------------------

            epsilon = linear_epsilon(
                global_step,
                eps_start,
                eps_end,
                eps_decay_steps
            )

            action = select_action(
                state,
                online_net,
                env.action_space,
                epsilon,
                device
            )

            # -------------------------------------------------
            # Ejecutar acción en el ambiente
            # -------------------------------------------------

            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            # -------------------------------------------------
            # Guardar transición
            #
            # IMPORTANTE:
            # almacenamos terminated independientemente
            # de truncated.
            # -------------------------------------------------

            replay_buffer.add(
                state,
                action,
                reward,
                next_state,
                terminated
            )

            episode_reward += reward
            episode_steps += 1
            max_distance = max(max_distance, float(next_state[0]) * 1000.0)

            # -------------------------------------------------
            # 4. Entrenamiento
            # -------------------------------------------------

            if (
                len(replay_buffer) >= max(learning_starts, batch_size)
                and global_step % train_frequency == 0
            ):

                (
                    states,
                    actions,
                    rewards,
                    next_states,
                    terminated_batch
                ) = replay_buffer.sample(batch_size)

                # Convertir minibatch a tensores

                states = torch.as_tensor(
                    states,
                    dtype=torch.float32,
                    device=device
                )

                actions = torch.as_tensor(
                    actions,
                    dtype=torch.long,
                    device=device
                )

                rewards = torch.as_tensor(
                    rewards,
                    dtype=torch.float32,
                    device=device
                )

                next_states = torch.as_tensor(
                    next_states,
                    dtype=torch.float32,
                    device=device
                )

                terminated_batch = torch.as_tensor(
                    terminated_batch,
                    dtype=torch.float32,
                    device=device
                )

                # -------------------------------------------------
                # 5. Calcular Q(s,a)
                # -------------------------------------------------

                q_values = online_net(states)

                current_q = q_values.gather(
                    1,
                    actions.unsqueeze(1)
                ).squeeze(1)

                # -------------------------------------------------
                # 6. Construir Target
                # -------------------------------------------------

                with torch.no_grad():

                    next_q_values = target_net(next_states)

                    max_next_q = next_q_values.max(
                        dim=1
                    ).values

                    target_q = (
                        rewards
                        + gamma
                        * (1.0 - terminated_batch)
                        * max_next_q
                    )

                # -------------------------------------------------
                # 7. Loss
                #
                # Smooth L1 = Huber Loss
                # -------------------------------------------------

                loss = F.smooth_l1_loss(
                    current_q,
                    target_q
                )

                # -------------------------------------------------
                # 8. Backpropagation
                # -------------------------------------------------

                optimizer.zero_grad()

                loss.backward()

                # Opcional pero útil para estabilidad
                torch.nn.utils.clip_grad_norm_(
                    online_net.parameters(),
                    max_norm=10.0
                )

                optimizer.step()

                losses.append(loss.item())

            # -------------------------------------------------
            # 9. Actualización periódica Target Network
            # -------------------------------------------------

            if (
                global_step > 0
                and
                global_step % target_update_frequency == 0
            ):

                target_net.load_state_dict(
                    online_net.state_dict()
                )

            # -------------------------------------------------
            # 10. Avanzar estado
            # -------------------------------------------------

            state = next_state
            global_step += 1

        # -----------------------------------------------------
        # Fin episodio
        # -----------------------------------------------------

        episode_rewards.append(episode_reward)

        avg_loss = (
            np.mean(losses)
            if len(losses) > 0
            else 0.0
        )

        episode_losses.append(avg_loss)
        history.append(dict(episode=episode+1, reward=float(episode_reward), episode_length=episode_steps, epsilon=float(epsilon), avg_loss=float(avg_loss), terminated=bool(terminated), truncated=bool(truncated), max_distance=float(max_distance)))

        # Permite a quien llama registrar el mejor checkpoint sin reentrenar
        if episode_callback is not None:
            episode_callback(episode, float(episode_reward), online_net)

        print(
            f"Episode: {episode + 1:4d} | "
            f"Reward: {episode_reward:8.2f} | "
            f"Epsilon: {epsilon:.3f} | "
            f"Loss: {avg_loss:.5f} | "
            f"Steps: {global_step}"
        )

    return {
        "rewards": episode_rewards,
        "losses": episode_losses,
        "history": history
    }