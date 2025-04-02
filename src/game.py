import copy
import json
import random

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


def check_abilities(player_card, other_cards, is_second_phase):
    """Check abilities of a card and calculate winnings."""
    result = {"stars": 0, "hearts": 0, "crystals": 0}

    # Check if player_card has abilities
    if "abilities" not in player_card or not player_card["abilities"]:
        return result

    for ability in get_card_abilities(player_card, is_second_phase):
        ability_winnings = check_ability(ability, player_card, other_cards, is_second_phase)
        if ability_winnings is not None:
            result["hearts"] += ability_winnings["hearts"]
            result["crystals"] += ability_winnings["crystals"]
            result["stars"] += ability_winnings["stars"]
            if ability_winnings["hearts"] > 0 or ability_winnings["crystals"] > 0 or ability_winnings["stars"] > 0:
                print(f"{player_card['name']} ability: {ability['condition']} {ability['condition_type']}"
                      f" won {ability_winnings['stars']} stars, {ability_winnings['hearts']} hearts, {ability_winnings['crystals']} crystals!")
    return result


def check_ability(ability, player_card, other_cards, is_second_phase):
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
    # Initialize the model
    model = MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation="relu",
                         random_state=1, max_iter=2000, warm_start=True)

    # Player AI will control
    player_index = 0

    # Load or create deck
    deck = cards_data

    # Initialize model with dummy data
    X_init = np.zeros((1, 170))  # Adjust size based on get_unified_state_features
    y_init = np.zeros(1)
    model.fit(X_init, y_init)

    ai_wins = 0
    for episode in range(episodes):
        if episode % 10 == 0:
            print(f"Starting episode {episode + 1}/{episodes}")

        # Deal cards
        players_hands, _ = deal_cards(deck)
        display_player_hands(players_hands, 'dealt')
        player_scores = [{"stars": 0, "hearts": 0, "crystals": 0} for _ in range(4)]
        drafted_hands = [[] for _ in range(4)]  # Keep track of drafted hands

        episode_memory = []  # Store (state, action, reward) tuples for training

        # --- Drafting Phase ---
        remaining_hands = copy.deepcopy(players_hands)
        for draft_round in range(7):
            for player_draft_index in range(4):
                current_hand = remaining_hands[player_draft_index]
                if not current_hand:
                    continue
                # Create game state for drafting
                drafting_state = get_game_state(
                    copy.deepcopy(remaining_hands),
                    copy.deepcopy(player_scores),
                    0,  # round num
                    [],  # played cards
                    player_draft_index,
                    draft_round,
                    copy.deepcopy(drafted_hands)
                )

                if player_draft_index == player_index:
                    card = ai_choose_card(current_hand, drafting_state, model, player_draft_index, True)
                else:
                    card = random.choice(current_hand)

                if not card:
                    continue
                # Store state and action for training
                if player_draft_index == player_index:
                    features = get_unified_state_features(drafting_state, player_index)
                    episode_memory.append((features, card, None))  # None reward for now

                drafted_hands[player_draft_index].append(card)
                current_hand.remove(card)
                remaining_hands[player_draft_index] = current_hand

            # Rotate hands
            rotated_hands = [remaining_hands[(player_draft_index - 1) % 4] for player_draft_index in range(4)]
            remaining_hands = rotated_hands

        display_player_hands(drafted_hands, 'drafted')
        # --- Playing Phase ---
        for round_num in range(7):
            num_players = 4
            starting_player = random.randint(0, num_players - 1)
            played_cards = []

            # Create game state
            game_state = get_game_state(
                copy.deepcopy(drafted_hands),
                copy.deepcopy(player_scores),
                round_num,
                played_cards,
                starting_player,
                7,  # draft round
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
                    episode_memory.append((features, card, None))  # None reward for now

                drafted_hands[current_player_index].remove(card)
                played_cards.append((current_player_index, card))

            # Determine round winner
            if played_cards:
                winner_index = max(played_cards, key=lambda x: x[1]["power"])[0]
                player_scores[winner_index]["stars"] += 1

                # Apply card abilities
                for current_player_index, card in played_cards:
                    other_cards = [c[1] for c in played_cards if c[0] != current_player_index]
                    winnings = check_abilities(card, other_cards, True)
                    if winnings:
                        player_scores[current_player_index]["hearts"] += winnings["hearts"]
                        player_scores[current_player_index]["crystals"] += winnings["crystals"]
                        player_scores[current_player_index]["stars"] += winnings["stars"]

        # Calculate final scores
        final_scores = calculate_final_scores(player_scores, drafted_hands)

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
                episode_memory[i] = (episode_memory[i][0], episode_memory[i][1], 1)  # Win reward
        else:
            for i in range(len(episode_memory)):
                episode_memory[i] = (episode_memory[i][0], episode_memory[i][1], 0)  # Loss reward

        # Train the model
        train_x = np.array([state for state, _, _ in episode_memory])
        train_y = np.array([reward for _, _, reward in episode_memory])
        model.partial_fit(train_x, train_y)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1} finished. AI win rate: {ai_wins / (episode + 1):.2f}")

    print(f"Training complete. AI won {ai_wins} out of {episodes} games ({ai_wins / episodes:.2%})")
    return model


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

    episodes = 1000
    if len(sys.argv) > 1:
        try:
            episodes = int(sys.argv[1])
        except ValueError:
            print(f"Invalid episodes value, using default: {episodes}")
    print(f"Training for {episodes} episodes")
    trained_model = train_and_play_game(episodes=episodes, cards_data=cards_data)
