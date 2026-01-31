"""Injury data scraper - fetches player injury information.

Uses free sources to get current injury status for EPL teams.
"""

import re
from datetime import datetime
from typing import Any, Optional

import httpx
import structlog
from bs4 import BeautifulSoup

logger = structlog.get_logger()

# Team name mappings from various sources to our standard names
TEAM_NAME_MAP = {
    # Premier Injuries / Transfermarkt variations
    "manchester united": "Manchester United",
    "man utd": "Manchester United",
    "manchester city": "Manchester City",
    "man city": "Manchester City",
    "liverpool": "Liverpool",
    "arsenal": "Arsenal",
    "chelsea": "Chelsea",
    "tottenham": "Tottenham Hotspur",
    "tottenham hotspur": "Tottenham Hotspur",
    "spurs": "Tottenham Hotspur",
    "newcastle": "Newcastle United",
    "newcastle united": "Newcastle United",
    "west ham": "West Ham United",
    "west ham united": "West Ham United",
    "aston villa": "Aston Villa",
    "brighton": "Brighton & Hove Albion",
    "brighton & hove albion": "Brighton & Hove Albion",
    "brentford": "Brentford",
    "crystal palace": "Crystal Palace",
    "fulham": "Fulham",
    "wolves": "Wolverhampton Wanderers",
    "wolverhampton": "Wolverhampton Wanderers",
    "wolverhampton wanderers": "Wolverhampton Wanderers",
    "bournemouth": "AFC Bournemouth",
    "afc bournemouth": "AFC Bournemouth",
    "nottingham forest": "Nottingham Forest",
    "nottm forest": "Nottingham Forest",
    "everton": "Everton",
    "luton": "Luton Town",
    "luton town": "Luton Town",
    "burnley": "Burnley",
    "sheffield united": "Sheffield United",
    "sheffield utd": "Sheffield United",
    "ipswich": "Ipswich Town",
    "ipswich town": "Ipswich Town",
    "leicester": "Leicester City",
    "leicester city": "Leicester City",
    "southampton": "Southampton",
}


class InjuryScraper:
    """Scrapes injury data from Premier Injuries website."""

    BASE_URL = "https://www.premierinjuries.com"

    def __init__(self):
        self.session = None

    async def get_all_injuries(self) -> dict[str, list[dict]]:
        """Get current injuries for all EPL teams.

        Returns:
            Dict mapping team name to list of injury dicts
        """
        injuries = {}

        async with httpx.AsyncClient(timeout=30) as client:
            try:
                response = await client.get(
                    f"{self.BASE_URL}/injury-table.php",
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    },
                )
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "html.parser")
                injuries = self._parse_injury_table(soup)

            except httpx.HTTPError as e:
                logger.warning(f"Failed to fetch injuries: {e}")
            except Exception as e:
                logger.error(f"Error parsing injuries: {e}")

        return injuries

    def _parse_injury_table(self, soup: BeautifulSoup) -> dict[str, list[dict]]:
        """Parse the injury table HTML."""
        injuries = {}

        # Find all team sections
        team_sections = soup.find_all("div", class_="team-injuries")

        for section in team_sections:
            team_name_elem = section.find("h2") or section.find("h3")
            if not team_name_elem:
                continue

            team_name = self._normalize_team_name(team_name_elem.text.strip())
            if not team_name:
                continue

            team_injuries = []

            # Find injury rows
            rows = section.find_all("tr", class_="injury-row") or section.find_all("tr")[1:]

            for row in rows:
                cells = row.find_all("td")
                if len(cells) >= 3:
                    injury_info = self._parse_injury_row(cells)
                    if injury_info:
                        team_injuries.append(injury_info)

            injuries[team_name] = team_injuries

        return injuries

    def _parse_injury_row(self, cells: list) -> Optional[dict]:
        """Parse a single injury row."""
        try:
            player_name = cells[0].text.strip()
            injury_type = cells[1].text.strip() if len(cells) > 1 else "Unknown"
            return_date = cells[2].text.strip() if len(cells) > 2 else "Unknown"
            status = cells[3].text.strip() if len(cells) > 3 else "Out"

            # Determine if key player (basic heuristic - could enhance with player data)
            is_key_player = self._is_key_player(player_name, status)

            return {
                "name": player_name,
                "type": injury_type,
                "return_date": return_date,
                "status": status,
                "is_key_player": is_key_player,
            }
        except Exception:
            return None

    def _is_key_player(self, player_name: str, status: str) -> bool:
        """Determine if player is likely a key player.

        This is a simplified heuristic - in production you'd use
        actual player value/rating data.
        """
        # Check for common indicators of important players
        status_lower = status.lower()
        if "doubt" in status_lower or "50%" in status:
            return True  # Doubt status usually only mentioned for key players

        return False

    def _normalize_team_name(self, raw_name: str) -> Optional[str]:
        """Normalize team name to our standard format."""
        cleaned = raw_name.lower().strip()
        return TEAM_NAME_MAP.get(cleaned)


class TransfermarktScraper:
    """Alternative scraper using Transfermarkt for injury data."""

    BASE_URL = "https://www.transfermarkt.com"

    # Transfermarkt team URL slugs
    TEAM_SLUGS = {
        "Manchester City": "manchester-city/verein/281",
        "Arsenal": "fc-arsenal/verein/11",
        "Liverpool": "fc-liverpool/verein/31",
        "Manchester United": "manchester-united/verein/985",
        "Tottenham Hotspur": "tottenham-hotspur/verein/148",
        "Chelsea": "fc-chelsea/verein/631",
        "Newcastle United": "newcastle-united/verein/762",
        "Brighton & Hove Albion": "brighton-amp-hove-albion/verein/1237",
        "West Ham United": "west-ham-united/verein/379",
        "Aston Villa": "aston-villa/verein/405",
        "AFC Bournemouth": "afc-bournemouth/verein/989",
        "Fulham": "fc-fulham/verein/931",
        "Crystal Palace": "crystal-palace/verein/873",
        "Brentford": "fc-brentford/verein/1148",
        "Wolverhampton Wanderers": "wolverhampton-wanderers/verein/543",
        "Nottingham Forest": "nottingham-forest/verein/703",
        "Everton": "fc-everton/verein/29",
        "Ipswich Town": "ipswich-town/verein/677",
        "Leicester City": "leicester-city/verein/1003",
        "Southampton": "fc-southampton/verein/180",
    }

    async def get_team_injuries(self, team_name: str) -> list[dict]:
        """Get injuries for a specific team from Transfermarkt."""
        slug = self.TEAM_SLUGS.get(team_name)
        if not slug:
            logger.warning(f"No Transfermarkt slug for {team_name}")
            return []

        injuries = []

        async with httpx.AsyncClient(timeout=30) as client:
            try:
                url = f"{self.BASE_URL}/{slug}/kader"
                response = await client.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    },
                )
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "html.parser")

                # Find players with injury icons
                injury_icons = soup.find_all("span", class_="verletzt-icon")
                for icon in injury_icons:
                    player_row = icon.find_parent("tr")
                    if player_row:
                        injury_info = self._parse_player_injury(player_row)
                        if injury_info:
                            injuries.append(injury_info)

            except httpx.HTTPError as e:
                logger.warning(f"Failed to fetch injuries for {team_name}: {e}")
            except Exception as e:
                logger.error(f"Error parsing injuries for {team_name}: {e}")

        return injuries

    def _parse_player_injury(self, row) -> Optional[dict]:
        """Parse player injury from table row."""
        try:
            name_cell = row.find("td", class_="hauptlink")
            player_name = name_cell.text.strip() if name_cell else "Unknown"

            # Get injury details from tooltip or injury column
            injury_cell = row.find("td", class_="zentriert")
            injury_type = "Injured"

            # Try to get market value as proxy for key player
            value_cell = row.find("td", class_="rechts")
            market_value = 0
            if value_cell:
                value_text = value_cell.text.strip()
                market_value = self._parse_market_value(value_text)

            return {
                "name": player_name,
                "type": injury_type,
                "return_date": "Unknown",
                "status": "Out",
                "is_key_player": market_value > 20_000_000,  # > £20m = key player
                "market_value": market_value,
            }
        except Exception:
            return None

    def _parse_market_value(self, value_text: str) -> int:
        """Parse market value string to integer."""
        try:
            # Remove currency symbols and parse
            cleaned = re.sub(r"[£€$,]", "", value_text.lower())
            if "m" in cleaned:
                return int(float(cleaned.replace("m", "")) * 1_000_000)
            elif "k" in cleaned:
                return int(float(cleaned.replace("k", "")) * 1_000)
            return int(cleaned) if cleaned.isdigit() else 0
        except Exception:
            return 0


async def get_injuries_for_team(team_name: str) -> dict[str, Any]:
    """Get injury data for a specific team.

    Tries multiple sources and combines results.

    Returns:
        Dict with injury info:
        {
            "players": [{"name": "...", "type": "...", "return_date": "..."}],
            "count": int,
            "key_players_out": int,
            "updated_at": datetime
        }
    """
    injuries = []
    key_players = 0

    # Try Premier Injuries first
    scraper = InjuryScraper()
    all_injuries = await scraper.get_all_injuries()

    if team_name in all_injuries:
        injuries = all_injuries[team_name]
    else:
        # Fallback to Transfermarkt
        tm_scraper = TransfermarktScraper()
        injuries = await tm_scraper.get_team_injuries(team_name)

    # Count key players
    key_players = sum(1 for inj in injuries if inj.get("is_key_player", False))

    return {
        "players": injuries,
        "count": len(injuries),
        "key_players_out": key_players,
        "updated_at": datetime.utcnow().isoformat(),
    }
