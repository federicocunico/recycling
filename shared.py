from enum import IntEnum

APP_PORT = 37012

months = {
    "0": "01_Gennaio",  # "Gennaio",
    "1": "02_Febbraio",  # "Febbraio",
    "2": "03_Marzo",  # "Marzo",
    "3": "04_Aprile",  # "Aprile",
    "4": "05_Maggio",  # "Maggio",
    "5": "06_Giugno",  # "Giugno",
    "6": "07_Luglio",  # "Luglio",
    "7": "08_Agosto",  # "Agosto",
    "8": "09_Settembre",  # "Settembre",
    "9": "10_Ottobre",  # "Ottobre",
    "10": "11_Novembre",  # "Novembre",
    "11": "12_Dicembre",  # "Dicembre",
}

days_by_months = {
    "0": 31,
    "1": 29,  # Always extract, eventually do not use.
    "2": 31,
    "3": 30,
    "4": 31,
    "5": 30,
    "6": 31,
    "7": 31,
    "8": 30,
    "9": 31,
    "10": 30,
    "11": 31,
}

keywords = ["umido", "secco", "plastica", "carta", "verde"]

weekdays = [
    "lunedì",
    "lunedi",
    "martedì",
    "martedi",
    "mercoledì",
    "mercoledi",
    "giovedì",
    "giovedi",
    "venerdì",
    "venerdi",
    "sabato",
    "domenica",
]
weekdays_forward_search = [
    "lunedì",
    "martedì",
    "mercoledì",
    "giovedì",
    "venerdì",
    "sabato",
    "domenica",
]
weekdays_backward_search = [
    "domenica",
    "sabato",
    "venerdì",
    "giovedì",
    "mercoledì",
    "martedì",
    "lunedì",
]



# class Waste(IntEnum):
#     PLASTICA = 0
#     UMIDO = 1
#     SECCO = 2
#     CARTA = 3
#     VERDE = 4
