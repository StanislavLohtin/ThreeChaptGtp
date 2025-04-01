import json
import random


def create_card_deck():
    """Creates a deck of 50 unique cards."""
    new_deck = []
    try:
        with open("cards.json", "r") as f:
            cards_data = json.load(f)
            for card in cards_data:
                new_deck.append(card)
    except FileNotFoundError:
        print(f"Error: File cards.json not found.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in cards.json .")
    return new_deck


def deal_cards(deck_to_deal, num_players=4, cards_per_player=8):
    """Deals cards to players."""
    hands = [[] for _ in range(num_players)]
    random.shuffle(deck_to_deal)

    for player_index in range(num_players):
        hands[player_index] = deck_to_deal[:cards_per_player]
        deck_to_deal = deck_to_deal[cards_per_player:]

    return hands, deck_to_deal


def draft_cards(hands, num_drafts=7):
    """Drafts cards between players."""
    hands_to_draft = [[] for _ in range(len(hands))]
    remaining_hands = hands.copy()

    for draft_round in range(num_drafts):
        for player_index in range(len(hands)):
            # get the current hand
            current_hand = remaining_hands[player_index]
            # select a card
            selected_card = random.choice(current_hand)
            # add the card to the drafted hand
            hands_to_draft[player_index].append(selected_card)
            # Remove the card from the current hand
            current_hand.remove(selected_card)
            # update the remaining hands.
            remaining_hands[player_index] = current_hand

        # Rotate the hands for the next round
        rotated_hands = [remaining_hands[(player_index - 1) % len(hands)] for player_index in
                         range(len(hands))]
        remaining_hands = rotated_hands

    return hands_to_draft


def display_player_hands(hands, description):
    for player_index, hand in enumerate(hands):
        print(f"Player {player_index + 1}'s {description} hand:")
        for card in hand:
            print(f"  {card['name']} (power: {card['power']}, Types: {card['types']})")
        print("-" * 20)


# Main game setup
deck = create_card_deck()
players_hands, remaining_deck = deal_cards(deck)
display_player_hands(players_hands, "dealt")
drafted_hands = draft_cards(players_hands)
display_player_hands(drafted_hands, "drafted")
