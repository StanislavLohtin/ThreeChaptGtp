import copy
import json
import random
import time
from collections import defaultdict

import numpy as np
from sklearn.neural_network import MLPRegressor


def deal_cards(deck, num_players=4, cards_per_player=8):
    """Deals cards to players."""
    hands = [[] for _ in range(num_players)]
    # Create a copy of the deck to avoid modifying the original
    shuffled_deck = deck.copy()
    random.shuffle(shuffled_deck)

    for player_index in range(num_players):
        hands[player_index] = shuffled_deck[:cards_per_player]
        shuffled_deck = shuffled_deck[cards_per_player:]

    return hands, shuffled_deck


def draft_cards(hands, num_drafts=7):
    """Drafts cards between players."""
    drafted_hands = [[] for _ in range(len(hands))]
    remaining_hands = copy.deepcopy(hands)  # Deep copy to avoid modifying original hands

    for draft_round in range(num_drafts):
        for player_index in range(len(hands)):
            current_hand = remaining_hands[player_index]
            if not current_hand:  # Check if hand is empty
                continue
            selected_card = random.choice(current_hand)
            drafted_hands[player_index].append(selected_card)
            current_hand.remove(selected_card)
            remaining_hands[player_index] = current_hand

        rotated_hands = [remaining_hands[(player_index - 1) % len(hands)] for player_index in range(len(hands))]
        remaining_hands = rotated_hands

    return drafted_hands


def display_player_hands(hands, description):
    """Displays the hands of each player with a description."""
    for player_index, hand in enumerate(hands):
        print(f"Player {player_index + 1}'s {description} hand:")
        for card in hand:
            print(f"  {card['name']} (power: {card['power']}, Types: {card['types']})")
        print("-" * 20)


def check_abilities(player_card, other_cards, is_second_phase, current_player_index=0):
    """Check abilities of a card and calculate winnings."""
    result = {"stars": 0, "hearts": 0, "crystals": 0}

    # Check if player_card has abilities
    if "abilities" not in player_card or not player_card["abilities"]:
        return result

    for ability in get_card_abilities(player_card, is_second_phase):
        ability_winnings = check_ability(ability, player_card, other_cards, is_second_phase, current_player_index)
        if ability_winnings is not None:
            if is_second_phase and get_cards_by_type(other_cards, "block_hearts_and_crystals"):
                ability_winnings["hearts"] = 0
                ability_winnings["crystals"] = 0
            result["hearts"] += ability_winnings["hearts"]
            result["crystals"] += ability_winnings["crystals"]
            result["stars"] += ability_winnings["stars"]
            if ability_winnings["hearts"] > 0 or ability_winnings["crystals"] > 0 or ability_winnings["stars"] > 0:
                print(f"{player_card['name']} ability: {ability['condition']} {ability['condition_type']}"
                      f" won {ability_winnings['stars']} stars, {ability_winnings['hearts']} hearts, {ability_winnings['crystals']} crystals!")
    return result


def check_ability(ability, player_card, other_cards, is_second_phase, current_player_index=0):
    """Check a specific ability and return winnings."""
    if ability["condition"] == "exists_one_other":
        # Return winnings if at least one card of the specified type exists
        return {"stars": ability["stars"], "hearts": ability["hearts"], "crystals": ability["crystals"]} if len(
            get_cards_by_type(other_cards, ability["condition_type"])) > 0 else {"stars": 0, "hearts": 0, "crystals": 0}

    if ability["condition"] == "for_each_other":
        # Calculate winnings for each matching card (excluding player's card)
        others_matched = len(get_cards_by_type(other_cards, ability["condition_type"]))
        return {"stars": ability["stars"] * others_matched, "hearts": ability["hearts"] * others_matched, "crystals": ability["crystals"] * others_matched}

    if ability["condition"] == "for_each":
        # Calculate winnings for each matching card (including player's card)
        others_matched = len(get_cards_by_type([player_card] + other_cards, ability["condition_type"]))
        return {"stars": ability["stars"] * others_matched, "hearts": ability["hearts"] * others_matched, "crystals": ability["crystals"] * others_matched}

    if ability["condition"] == "with":
        # Check if a specific card (by power) exists
        card_found = 1 if get_card_by_power(other_cards, ability["condition_type"]) is not None else 0
        return {"stars": ability["stars"] * card_found, "hearts": ability["hearts"] * card_found, "crystals": ability["crystals"] * card_found}

    if ability["condition"] == "for_each_in_range":
        ranges_string = ability["condition_type"].split(',')
        cards_found = len(get_cards_in_power_range(other_cards, int(ranges_string[0]), int(ranges_string[1])))
        return {"stars": ability["stars"] * cards_found, "hearts": ability["hearts"] * cards_found, "crystals": ability["crystals"] * cards_found}

    if ability["condition"] == "if_you_win":
        won = 1 if max(card["power"] for card in other_cards) < player_card["power"] else 0
        return {"stars": ability["stars"] * won, "hearts": ability["hearts"] * won, "crystals": ability["crystals"] * won}

    if ability["condition"] == "if_you_win_with":
        won = 1 if max(card["power"] for card in other_cards) < player_card["power"] else 0
        won = won * (1 if get_card_by_power(other_cards, ability["condition_type"]) is not None else 0)
        return {"stars": ability["stars"] * won, "hearts": ability["hearts"] * won, "crystals": ability["crystals"] * won}

    if ability["condition"] == "if_you_win_without":
        won = 1 if max(card["power"] for card in other_cards) < player_card["power"] else 0
        won = won * (1 if get_card_by_power(other_cards, ability["condition_type"]) is None else 0)
        return {"stars": ability["stars"] * won, "hearts": ability["hearts"] * won, "crystals": ability["crystals"] * won}

    if ability["condition"] == "if_weakest":
        has_lowest = 1 if min(card["power"] for card in other_cards) > player_card["power"] else 0
        return {"stars": ability["stars"] * has_lowest, "hearts": ability["hearts"] * has_lowest, "crystals": ability["crystals"] * has_lowest}

    if ability["condition"] == "if_last":
        last = 1 if current_player_index == 3 else 0
        return {"stars": ability["stars"] * last, "hearts": ability["hearts"] * last, "crystals": ability["crystals"] * last}

    return {"stars": 0, "hearts": 0, "crystals": 0}


def calculate_final_scores(player_scores, player_hands):
    """Calculates final scores and determines the winner."""
    for player_index, score in enumerate(player_scores):
        print(f"Checking final score for player {player_index + 1}:")
        for card in player_hands[player_index]:
            winnings = check_abilities(card, [c for c in player_hands[player_index] if c != card], False)
            score["stars"] += winnings["stars"]
            score["hearts"] += winnings["hearts"]
            score["crystals"] += winnings["crystals"]
        print()

    highest_crystals = max(score["crystals"] for score in player_scores)
    crystal_winners = [i + 1 for i, score in enumerate(player_scores) if score["crystals"] == highest_crystals]  # Fixed index display

    if len(crystal_winners) > 0:
        print(f"Players {crystal_winners} gain 4 extra Crystals for having the most crystals.")
        for winner_index in crystal_winners:
            player_scores[winner_index - 1]["crystals"] += 4  # Adjust index back

    final_scores = []
    for player_index, score in enumerate(player_scores):
        total_points = (score["stars"] * 2) + score["hearts"] + (score["crystals"] // 2)
        final_scores.append((player_index, total_points))

    highest_score = max(score for _, score in final_scores)
    winners = [player_index for player_index, score in final_scores if score == highest_score]

    if len(winners) == 1:
        print(f"Player {winners[0] + 1} wins the game!")
    else:
        print(f"Tie between players: {[w + 1 for w in winners]}")  # Fix display of player numbers
        tiebreaker_winners = []
        highest_power = -1
        for winner in winners:
            player_max_power = -1
            for card in player_hands[winner]:
                if card['power'] > player_max_power:
                    player_max_power = card['power']
            if player_max_power > highest_power:
                highest_power = player_max_power
                tiebreaker_winners = [winner]
            elif player_max_power == highest_power:
                tiebreaker_winners.append(winner)

        if len(tiebreaker_winners) == 1:
            print(f"Player {tiebreaker_winners[0] + 1} wins the tiebreaker!")
        else:
            print(f"Still a tie between players: {[w + 1 for w in tiebreaker_winners]}")  # Fix display

    print("\n--- Final Results ---")
    for player_index, score in enumerate(player_scores):
        print(
            f"Player {player_index + 1}: Stars = {score['stars']}, Hearts = {score['hearts']}, Crystals = {score['crystals']}, Final Points = {final_scores[player_index][1]}")

    return final_scores


def get_card_abilities(card, is_second_phase):
    """Get all applicable abilities for a card based on the game phase."""
    result = []
    if "abilities" in card:
        for ability in card["abilities"]:
            # Check if ability triggers in current phase
            if ability["trigger"] == "both" or (is_second_phase and ability["trigger"] == "second") or (not is_second_phase and ability["trigger"] == "third"):
                result.append(ability)
    return result


def get_cards_by_type(cards, type_name):
    """Find all cards of a specific type."""
    return [card for card in cards if type_name in card["types"]]


def get_cards_in_power_range(cards, min_power, max_power):
    """Find a card with a specific power value."""
    return [card for card in cards if (card["power"] >= min_power) and card["power"] <= max_power]


def get_card_by_power(cards, power):
    """Find a card with a specific power value."""
    return next((card for card in cards if card["power"] == power), None)


def get_game_state(player_hands, player_scores, round_num, played_cards, current_player_index, draft_round, drafted_hands):
    """Create a unified game state representation for both drafting and playing."""
    return {
        "player_hands": player_hands,
        "player_scores": player_scores,
        "round_num": round_num,
        "played_cards": played_cards,
        "current_player_index": current_player_index,
        "draft_round": draft_round,  # Add draft round
        "drafted_hands": drafted_hands,  # add drafted hands
    }


def get_unified_state_features(game_state, player_index):
    """Extract features from the game state for AI decision making, combining drafting and playing."""
    features = []

    # --- Drafting Phase Features ---
    # Add draft round
    features.append(game_state["draft_round"])

    # Add features for cards in current hand
    if game_state["player_hands"] and len(game_state["player_hands"]) > player_index:
        for card in game_state["player_hands"][player_index]:
            features.append(card["power"])  # Card power
            features.extend(encode_card_types(card))  # Encode card types

    # Add features for drafted cards so far
    if game_state["drafted_hands"] and len(game_state["drafted_hands"]) > player_index:
        for card in game_state["drafted_hands"][player_index]:
            features.append(card["power"])
            features.extend(encode_card_types(card))

    # --- Playing Phase Features ---
    # Add score features
    if game_state["player_scores"] and len(game_state["player_scores"]) > player_index:
        features.extend([
            game_state["player_scores"][player_index]["stars"],
            game_state["player_scores"][player_index]["hearts"],
            game_state["player_scores"][player_index]["crystals"]
        ])

    # Add round number
    features.append(game_state["round_num"])

    # Add played cards
    for _, card in game_state["played_cards"]:
        features.append(card["power"])  # Power of played card
        features.extend(encode_card_types(card))

    # Add current player index
    features.append(game_state["current_player_index"])

    # Pad features
    while len(features) < 170:
        features.append(0)
    return features


def encode_card_types(card):
    """Encodes card types into a numerical representation (one-hot)."""
    all_types = ['Monster', 'Tier', 'Boesewicht', 'Kind', 'Wunderland', 'Zauber', 'Zwerg', 'Krone']  # All possible types
    encoded_types = [1 if card_type in card['types'] else 0 for card_type in all_types]
    return encoded_types


def ai_choose_card(player_hand, game_state, model, player_index, is_drafting):
    """AI selects a card to draft or play using a trained model."""
    if not player_hand:
        return None

    q_values = []
    for i, card in enumerate(player_hand):
        # Create a hypothetical next state
        next_game_state = copy.deepcopy(game_state)
        if is_drafting:
            next_game_state["drafted_hands"] = copy.deepcopy(game_state["drafted_hands"])
            next_game_state["drafted_hands"][player_index] = copy.deepcopy(next_game_state["drafted_hands"][player_index])
            next_game_state["drafted_hands"][player_index].append(card)
            temp_hand = player_hand.copy()
            temp_hand.pop(i)
            next_game_state["player_hands"] = list(next_game_state["player_hands"])
            next_game_state["player_hands"][player_index] = temp_hand
        else:
            next_game_state["played_cards"] = copy.deepcopy(game_state["played_cards"])
            next_game_state["played_cards"].append((player_index, card))
            next_game_state["player_hands"] = list(next_game_state["player_hands"])
            next_game_state["player_hands"][player_index] = list(next_game_state["player_hands"][player_index])
            next_game_state["player_hands"][player_index].remove(card)

        features = get_unified_state_features(next_game_state, player_index)
        try:
            prediction = model.predict([features])[0]
            q_values.append(prediction)
        except:
            q_values.append(random.random())

    # Choose card with the highest Q-value
    best_card_index = np.argmax(q_values)
    return player_hand[best_card_index] if player_hand else None


def train_and_play_game(episodes=1000, cards_data=None):
    """Trains and plays the game using a single model for both drafting and playing."""
    # Load previous training data, if available
    previous_model_data = load_training_data()

    # Initialize the model
    if previous_model_data:
        print("Loading model from previous training data...")
        model = MLPRegressor(hidden_layer_sizes=previous_model_data['hidden_layer_sizes'], activation="relu",
                             random_state=1, max_iter=2000, warm_start=True)
        model.coefs_ = previous_model_data['coefs_']
        model.intercepts_ = previous_model_data['intercepts_']
        if 'n_layers_' in previous_model_data:
            model.n_layers_ = previous_model_data['n_layers_']
        if 'out_activation_' in previous_model_data:
            model.out_activation_ = previous_model_data['out_activation_']
        if 't_' in previous_model_data:
            model.t_ = previous_model_data['t_']
        if 'loss_curve_' in previous_model_data:
            model.loss_curve_ = previous_model_data['loss_curve_']
        if 'best_loss_' in previous_model_data:
            model.best_loss_ = previous_model_data['best_loss_']
        if '_no_improvement_count' in previous_model_data:
            model._no_improvement_count = previous_model_data['_no_improvement_count']
    else:
        print("No previous training data found. Training from scratch.")
        model = MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation="relu",
                             random_state=1, max_iter=2000, warm_start=True)

    # Player AI will control
    player_index = 0

    # Load or create deck
    deck = cards_data

    # Initialize model with dummy data
    X_init = np.zeros((1, 170))
    y_init = np.zeros(1)
    if not previous_model_data:
        model.fit(X_init, y_init)  # Only fit on dummy if no previous model data

    ai_wins = 0
    player_points = []
    card_draft_counts = defaultdict(int)
    card_win_counts = defaultdict(int)
    all_cards = set()

    for episode in range(episodes):
        if episode % 10 == 0:
            print(f"Starting episode {episode + 1}/{episodes}")

        # Deal cards
        players_hands, _ = deal_cards(deck)
        # display_player_hands(players_hands, 'dealt')
        player_scores = [{"stars": 0, "hearts": 0, "crystals": 0} for _ in range(4)]
        drafted_hands = [[] for _ in range(4)]

        episode_memory = []

        # --- Drafting Phase ---
        remaining_hands = copy.deepcopy(players_hands)
        for draft_round in range(7):
            for player_draft_index in range(4):
                current_hand = remaining_hands[player_draft_index]
                if not current_hand:
                    continue
                drafting_state = get_game_state(
                    copy.deepcopy(remaining_hands),
                    copy.deepcopy(player_scores),
                    0,
                    [],
                    player_draft_index,
                    draft_round,
                    copy.deepcopy(drafted_hands)
                )

                if player_draft_index == player_index:
                    card = ai_choose_card(current_hand, drafting_state, model, player_draft_index, True)
                    card_draft_counts[card["name"]] += 1
                    all_cards.add(card["name"])
                else:
                    card = random.choice(current_hand)

                if not card:
                    continue
                if player_draft_index == player_index:
                    features = get_unified_state_features(drafting_state, player_index)
                    episode_memory.append((features, card, None))

                drafted_hands[player_draft_index].append(card)
                current_hand.remove(card)
                remaining_hands[player_draft_index] = current_hand

            rotated_hands = [remaining_hands[(player_draft_index - 1) % 4] for player_draft_index in range(4)]
            remaining_hands = rotated_hands

        # display_player_hands(drafted_hands, 'drafted')
        hands_after_draft = copy.deepcopy(drafted_hands)

        # --- Playing Phase ---
        for round_num in range(7):
            num_players = 4
            starting_player = random.randint(0, num_players - 1)
            played_cards = []

            game_state = get_game_state(
                copy.deepcopy(drafted_hands),
                copy.deepcopy(player_scores),
                round_num,
                played_cards,
                starting_player,
                7,
                copy.deepcopy(drafted_hands)
            )
            for i in range(num_players):
                current_player_index = (starting_player + i) % num_players
                current_player_hand = drafted_hands[current_player_index]

                if not current_player_hand:
                    continue

                # Choose card
                if current_player_index == player_index:
                    card = ai_choose_card(current_player_hand, game_state, model, current_player_index, False)
                else:
                    card = random.choice(current_player_hand)

                if not card:
                    continue

                # Store state and action
                if current_player_index == player_index:
                    features = get_unified_state_features(game_state, player_index)
                    episode_memory.append((features, card, None))

                drafted_hands[current_player_index].remove(card)
                played_cards.append((current_player_index, card))

            # Determine round winner
            if played_cards:
                winner_index = max(played_cards, key=lambda x: x[1]["power"])[0]
                player_scores[winner_index]["stars"] += 1

                # Apply card abilities
                for current_player_index, card in played_cards:
                    other_cards = [c[1] for c in played_cards if c[0] != current_player_index]
                    winnings = check_abilities(card, other_cards, True, current_player_index)
                    if winnings:
                        player_scores[current_player_index]["hearts"] += winnings["hearts"]
                        player_scores[current_player_index]["crystals"] += winnings["crystals"]
                        player_scores[current_player_index]["stars"] += winnings["stars"]

        # Calculate final scores
        final_scores = calculate_final_scores(player_scores, hands_after_draft)
        player_points.append(final_scores[player_index][1])

        # Determine winner and assign rewards
        ai_won = False
        ai_final_score = final_scores[player_index][1]
        for p_idx, score in final_scores:
            if p_idx != player_index and score > ai_final_score:
                ai_won = False
                break
        else:
            ai_won = True

        if ai_won:
            ai_wins += 1
            for i in range(len(episode_memory)):
                episode_memory[i] = (episode_memory[i][0], episode_memory[i][1], 1)
                if episode_memory[i][1] is not None:
                    card_win_counts[episode_memory[i][1]["name"]] += 1
        else:
            for i in range(len(episode_memory)):
                episode_memory[i] = (episode_memory[i][0], episode_memory[i][1], 0)

        # Train the model
        train_x = np.array([state for state, _, _ in episode_memory])
        train_y = np.array([reward for _, _, reward in episode_memory])
        model.partial_fit(train_x, train_y)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1} finished. AI win rate: {ai_wins / (episode + 1):.2f}")

    # Save the trained model
    save_training_data(model, ai_wins, player_points)
    return model, ai_wins, player_points, card_draft_counts, card_win_counts, all_cards


def create_report(ai_wins, player_points, card_draft_counts, card_win_counts, all_cards, episodes, deck):
    """Generates a report summarizing the AI's performance."""
    report = f"Training completed over {episodes} episodes.\n"
    report += f"AI won {ai_wins} out of {episodes} games ({ai_wins / episodes:.2%})\n\n"

    report += "Player Points:\n"
    report += f"  Average: {np.mean(player_points):.2f}\n"
    report += f"  Median: {np.median(player_points):.2f}\n"
    report += f"  Maximum: {np.max(player_points)}\n\n"

    report += "Card Drafting Analysis:\n"
    report += "----------------------\n"
    card_data = []
    for card_name in all_cards:
        draft_count = card_draft_counts.get(card_name, 0)
        win_count = card_win_counts.get(card_name, 0)
        win_rate = (win_count / draft_count) if draft_count > 0 else 0
        card_power = 0
        for card in deck:
            if card['name'] == card_name:
                card_power = card['power']
                break
        card_data.append({
            "name": card_name,
            "power": card_power,
            "draft_probability": draft_count / episodes,
            "draft_count": draft_count,
            "win_rate": win_rate
        })

    # Sort cards by win rate
    sorted_card_data = sorted(card_data, key=lambda x: x["win_rate"], reverse=True)
    for card in sorted_card_data:
        report += f"  {card['name']} (Power: {card['power']}): Drafted {card['draft_count']} times, Draft probability: {card['draft_probability']:.2f}, Win Rate: {card['win_rate']:.2f}%\n"

    return report


def save_training_data(model, ai_wins, player_points):
    """Saves the training data to a JSON file."""
    model_data = {
        'coefs_': [coef.tolist() for coef in model.coefs_],
        'intercepts_': [intercept.tolist() for intercept in model.intercepts_],
        'ai_wins': ai_wins,
        'player_points': player_points,
        'n_layers_': getattr(model, 'n_layers_', 0),  # Save n_layers_ if it exists
        'hidden_layer_sizes': model.hidden_layer_sizes,  # save the hidden layer sizes
        'out_activation_': getattr(model, 'out_activation_', 'relu'),
        't_': getattr(model, 't_', 0),
        'loss_curve_': getattr(model, 'loss_curve_', []),
        'best_loss_': getattr(model, 'best_loss_', None),
        '_no_improvement_count': getattr(model, '_no_improvement_count', 0),
    }
    try:
        with open("training_data.json", "w") as f:
            json.dump(model_data, f)
        print("Training data saved to training_data.json")
    except Exception as e:
        print(f"Error saving training data: {e}")


def load_training_data():
    """Loads the training data from a JSON file."""
    try:
        with open("training_data.json", "r") as f:
            model_data = json.load(f)
            # Convert lists back to numpy arrays
            model_data['coefs_'] = [np.array(coef) for coef in model_data['coefs_']]
            model_data['intercepts_'] = [np.array(intercept) for intercept in model_data['intercepts_']]
            return model_data
    except FileNotFoundError:
        print("Training data file not found. Starting from scratch.")
        return None  # Return None to indicate no previous data
    except json.JSONDecodeError:
        print("Error decoding JSON. Starting from scratch.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}. Starting from scratch.")
        return None


# Add this block for testing
if __name__ == "__main__":
    # For testing
    import sys

    try:
        with open("cards.json", "r") as f:
            cards_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Creating cards from scratch...")
        cards_data = []

    episodes = 10000
    if len(sys.argv) > 1:
        try:
            episodes = int(sys.argv[1])
        except ValueError:
            print(f"Invalid episodes value, using default: {episodes}")
    print(f"Training for {episodes} episodes")
    trained_model, ai_wins, player_points, card_draft_counts, card_win_counts, all_cards = train_and_play_game(episodes=episodes, cards_data=cards_data)
    print(f"Trained model: {trained_model}")
    report = create_report(ai_wins, player_points, card_draft_counts, card_win_counts, all_cards, episodes, cards_data)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"reports/report_{timestamp}_X{episodes}.txt"
    with open(filename, "w") as f:
        f.write(report)
    print(f"Report saved to {filename}")
