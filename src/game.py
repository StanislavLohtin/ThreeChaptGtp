import copy
import json
import random


def create_card_deck():
    """Creates a deck of cards from cards.json."""
    deck = []
    try:
        with open("cards.json", "r") as f:
            cards_data = json.load(f)
            for card in cards_data:
                deck.append(card)
    except FileNotFoundError:
        print("Error: File cards.json not found.")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in cards.json.")
    return deck


def deal_cards(deck, num_players=4, cards_per_player=8):
    """Deals cards to players."""
    hands = [[] for _ in range(num_players)]
    random.shuffle(deck)

    for player_index in range(num_players):
        hands[player_index] = deck[:cards_per_player]
        deck = deck[cards_per_player:]

    return hands, deck


def draft_cards(hands, num_drafts=7):
    """Drafts cards between players."""
    drafted_hands = [[] for _ in range(len(hands))]
    remaining_hands = hands.copy()

    for draft_round in range(num_drafts):
        for player_index in range(len(hands)):
            current_hand = remaining_hands[player_index]
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


def play_round(player_hands, player_scores, round_num):
    """Plays a single round of the game."""
    num_players = len(player_hands)
    starting_player = random.randint(0, num_players - 1)
    played_cards = []

    print(f"\n--- Round {round_num + 1} ---")

    for i in range(num_players):
        player_index = (starting_player + i) % num_players
        card = player_hands[player_index].pop(0)  # Play the first card
        played_cards.append((player_index, card))
        print(f"Player {player_index + 1} played: {card['name']} (power: {card['power']})")

    # Determine the winner
    winner_index = max(played_cards, key=lambda x: x[1]['power'])[0]
    player_scores[winner_index]["stars"] += 1
    print(f"Player {winner_index + 1} wins the round!")

    # Calculate points for other players
    for player_index, card in played_cards:
        other_cards = [c[1] for c in played_cards if c[0] != player_index]
        winnings = check_abilities(card, other_cards, True)
        if winnings is not None:
            player_scores[player_index]["hearts"] += winnings["hearts"]
            player_scores[player_index]["crystals"] += winnings["crystals"]
            player_scores[player_index]["stars"] += winnings["stars"]


def check_abilities(player_card, other_cards, is_second_phase):
    result = {"stars": 0, "hearts": 0, "crystals": 0}
    for ability in get_card_abilities(player_card, is_second_phase):
        ability_winnings = check_ability(ability, player_card, other_cards, is_second_phase)
        if ability_winnings is not None:
            result["hearts"] += ability_winnings["hearts"]
            result["crystals"] += ability_winnings["crystals"]
            result["stars"] += ability_winnings["stars"]
            if ability_winnings["hearts"] > 0 or ability_winnings["crystals"] > 0 or ability_winnings["stars"] > 0:
                print(
                    f"{player_card['name']} ability: {ability['condition']} {ability['condition_type']}"
                    f" won {ability_winnings['stars']} stars, {ability_winnings['hearts']} hearts, {ability_winnings['crystals']} crystals!")
    return result


def check_ability(ability, player_card, other_cards, is_second_phase):
    if ability["condition"] == "exists_one_other":
        return {"stars": ability["stars"], "hearts": ability["hearts"], "crystals": ability["crystals"]} if len(
            get_cards_by_type(other_cards, ability["condition_type"])) > 0 else None
    if ability["condition"] == "for_each_other":
        others_matched = len(get_cards_by_type(other_cards, ability["condition_type"]))
        return {"stars": ability["stars"] * others_matched, "hearts": ability["hearts"] * others_matched, "crystals": ability["crystals"] * others_matched}
    if ability["condition"] == "for_each":
        others_matched = len(get_cards_by_type([player_card] + other_cards, ability["condition_type"]))
        return {"stars": ability["stars"] * others_matched, "hearts": ability["hearts"] * others_matched, "crystals": ability["crystals"] * others_matched}
    if ability["condition"] == "with":
        card_found = 1 if get_card_by_power(other_cards, ability["condition_type"]) is not None else 0
        return {"stars": ability["stars"] * card_found, "hearts": ability["hearts"] * card_found, "crystals": ability["crystals"] * card_found}
    return None


def calculate_final_scores(player_scores, player_hands):
    """Calculates final scores and determines the winner."""
    for player_index, score in enumerate(player_scores):
        print(f"Checking final score for player {player_index + 1}:")
        for card in player_hands[player_index]:
            winnigs = check_abilities(card, [c for c in player_hands[player_index] if c != card], False)
            score["stars"] += winnigs["stars"]
            score["hearts"] += winnigs["hearts"]
            score["crystals"] += winnigs["crystals"]
        print()

    # Find all players with the highest crystal count:
    highest_crystals = max(score["crystals"] for score in player_scores)
    crystal_winners = [i for i, score in enumerate(player_scores) if score["crystals"] == highest_crystals]

    if len(crystal_winners) > 0:
        print(f"Players {crystal_winners} gain 4 extra Crystals for having the most crystals.")
        for winner_index in crystal_winners:
            player_scores[winner_index]["crystals"] += 4

    # Calculate total points
    final_scores = []
    for player_index, score in enumerate(player_scores):
        total_points = (score["stars"] * 2) + score["hearts"] + (score["crystals"] // 2)
        final_scores.append((player_index, total_points))

    # Determine the winner(s)
    highest_score = max(score for _, score in final_scores)
    winners = [player_index for player_index, score in final_scores if score == highest_score]

    if len(winners) == 1:
        print(f"Player {winners[0] + 1} wins the game!")
    else:
        print(f"Tie between players: {winners}")
        # Tiebreaker: Highest power card
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
            print(f"Still a tie between players: {tiebreaker_winners}")

    # Log final scores
    print("\n--- Final Results ---")
    for player_index, score in enumerate(player_scores):
        print(
            f"Player {player_index + 1}: Stars = {score['stars']}, Hearts = {score['hearts']}, Crystals = {score['crystals']}, Final Points = {final_scores[player_index][1]}")


def play_game(player_hands):
    """Plays the main game loop."""
    starting_hands = copy.deepcopy(player_hands)
    num_rounds = len(player_hands[0])  # 7
    num_players = len(player_hands)
    player_scores = [{"stars": 0, "hearts": 0, "crystals": 0} for _ in range(num_players)]

    for round_num in range(num_rounds):
        play_round(player_hands, player_scores, round_num)

    calculate_final_scores(player_scores, starting_hands)


def get_card_abilities(card, is_second_phase):
    result = []
    if "abilities" in card:  # Check if the card has an abilities field.
        for ability in card["abilities"]:
            if ability["trigger"] == "both" or (
                    ability["trigger"] == "second" if is_second_phase else ability["trigger"] == "third"):
                result.append(ability)
    return result


def get_cards_by_type(cards, type):
    result = []
    for card in cards:
        for card_type in card["types"]:
            if card_type == type:
                result.append(card)
    return result


def get_card_by_power(cards, power):
    for card in cards:
        if card["power"] == power:
            return card
    return None


# Main game setup
deck = create_card_deck()
if deck:
    players_hands, remaining_deck = deal_cards(deck)
    display_player_hands(players_hands, "dealt")
    drafted_hands = draft_cards(players_hands)
    display_player_hands(drafted_hands, "drafted")
    play_game(drafted_hands)
