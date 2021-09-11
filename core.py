import json
import os
from enum import IntEnum
from datetime import datetime, timedelta
from shared import months
from typing import List, Optional


class WasteCalendar:
    def __init__(self, year: int, data_file: Optional[str] = None) -> None:
        assert year is not None, "Provide what year is it"
        self.year = year
        self.data = self.__set(year, data_file)
        self._debug_mode = False

    def __set(self, year, data_file):
        if not year and not data_file:
            raise RuntimeError("Provide at least one argument to the function")

        if not data_file:
            data_file = f"result_{year}.json"

        if not os.path.isfile(data_file):
            raise FileNotFoundError(f"Unable to open file: {data_file}")

        with open(data_file, "r") as fp:
            data = json.load(fp)
        return data

    def set(self, year: int):
        self.data = self.__set(year, None)
        self.year = year

    def _debug(self, s):
        if self._debug_mode:
            print(s)

    def get_data(self, today: Optional[datetime] = None) -> List[str]:
        if not today:
            today = datetime.today()
        
        if self.year != today.year:
            raise RuntimeError(f"Unexpected year! Got requested year {today.year} but I only have the data for year {self.year}")

        if today.month == 12 and today.day == 31:
            # l'ulitmo dell'anno, non ho il calendario dell'anno succesivo
            raise RuntimeError("I do not have such data")


        self._debug(f"Giorno di OGGI: {today.strftime('%d %B, %Y')}")
        tomorrow = today + timedelta(days=1)
        self._debug(f"Giorno di DOMANI: {tomorrow.strftime('%d %B, %Y')}")

        # Mese e giorno di domani
        month = tomorrow.month
        day = str(tomorrow.day)

        self._debug(f"Mese in valore {month}")
        self._debug(f"Giorno in valore {day}")

        # Nome comune del mese
        month_str = months[str(month - 1)]
        self._debug(f"Mese dai dati {month_str}")

        # Giorni del mese
        days = self.data[month_str]
        # Cosa raccolgono domani (i.e. cosa devi metter fuori oggi)
        waste_collection = days[day]
        self._debug(f"Dati per domani: {waste_collection}")

        return waste_collection


def __test():
    calendar = WasteCalendar(2021)
    # calendar._debug_mode = True
    data = calendar.get_data()
    print("Tomorrow: ", data)

    year = 2021
    month = 3
    day = 16
    requested_str = f"{year}-{month}-{day}"
    requested = datetime.strptime(requested_str, "%Y-%m-%d")

    data = calendar.get_data(requested)
    print(requested_str, ":", data)

if __name__ == "__main__":
    __test()
