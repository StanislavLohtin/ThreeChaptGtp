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


def checks_abilities(player_card, other_cards):
    ability = get_card_ability(player_card, True)
    if ability is None:
        return None
    if ability["condition"] == "exists_one_other":
        return {"stars": ability["stars"], "hearts": ability["hearts"], "crystals": ability["crystals"]} if len(
            get_cards_by_type(other_cards, "Krone")) > 0 else None
    return None


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
        winnings = checks_abilities(card, other_cards)
        if winnings is not None:
            player_scores[player_index]["hearts"] += winnings["hearts"]
            player_scores[player_index]["crystals"] += winnings["crystals"]
            player_scores[player_index]["stars"] += winnings["stars"]


def play_game(player_hands):
    """Plays the main game loop."""
    num_rounds = len(player_hands[0])  # 7
    num_players = len(player_hands)
    player_scores = [{"stars": 0, "hearts": 0, "crystals": 0} for _ in range(num_players)]

    for round_num in range(num_rounds):
        play_round(player_hands, player_scores, round_num)

    # Display final scores
    print("\n--- Final Scores ---")
    for player_index, score in enumerate(player_scores):
        print(
            f"Player {player_index + 1}: Stars = {score['stars']}, hearts = {score['hearts']}, crystals = {score['crystals']}")


def get_card_ability(card, is_second_phase):
    if "abilities" in card:  # Check if the card has an abilities field.
        for ability in card["abilities"]:
            if ability["trigger"] == "both" or (
                    ability["trigger"] == "second" if is_second_phase else ability["trigger"] == "third"):
                return ability
    return None


def get_cards_by_type(cards, type):
    result = []
    for card in cards:
        for card_type in card["types"]:
            if card_type == type:
                result.append(card)
    return result


# Main game setup
deck = create_card_deck()
if deck:
    players_hands, remaining_deck = deal_cards(deck)
    drafted_hands = draft_cards(players_hands)
    play_game(drafted_hands)
