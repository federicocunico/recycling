#!/usr/bin/env python
# pylint: disable=C0116,W0613

import logging
from datetime import time, datetime, timedelta
from typing import Dict
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext
from config import TELEGRAM_SECRET_TOKEN
from core import WasteCalendar

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)

waste_calenders: Dict[str, WasteCalendar] = {}

# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
# Best practice would be to replace context with an underscore,
# since context is an unused local variable.
# This being an example and not having context present confusing beginners,
# we decided to have it present as context.
def start(update: Update, context: CallbackContext) -> None:
    """Sends explanation on how to use the bot."""
    s = (
        "I really hope you live in Bussolengo (VR) "
        + "and in the B zone of it, otherwise this won't be useful (:"
    )
    update.message.reply_text(s)
    help(update)


def help(update: Update):
    commands = [
        "/set <year>  ->  to start the bot (daily, 20:00)",
        "/stop  ->  to stop the bot",
        "/today  ->  to get today waste collection",
        "/help  ->  to show the commands",
    ]
    update.message.reply_text("\n".join(commands))


def waste_collection(context: CallbackContext) -> None:
    # """Look for waste."""
    try:
        chat_id = context.job.context["chat_id"]
        update = context.job.context["update"]
    except AttributeError:
        chat_id = context["chat_id"]
        update = context["update"]

    # logger.info(chat_id)
    if chat_id not in waste_calenders:
        waste_calenders[chat_id] = WasteCalendar(datetime.now().year)
    data = waste_calenders[chat_id].get_data()
    raccolta = data["raccolta"]
    giorno = data["giorno"]
    if not raccolta:
        msg = f"Domani mattina, {giorno}, non passano a raccogliere nulla."
    else:
        msg = (
            f"Domani mattina, {giorno}, passano a raccogliere "
            + ", ".join(raccolta)
            + "."
        )

    update.message.reply_text(msg)

    # context.bot.send_message(job.context, text=str(data))


def remove_job_if_exists(name: str, context: CallbackContext) -> bool:
    """Remove job with given name. Returns whether job was removed."""
    current_jobs = context.job_queue.get_jobs_by_name(name)
    if not current_jobs:
        return False
    for job in current_jobs:
        job.schedule_removal()
    return True


def waste_set(update: Update, context: CallbackContext) -> None:
    chat_id = update.message.chat_id
    try:
        # args[0] should contain the year
        year = int(context.args[0])
        waste_calenders[chat_id] = WasteCalendar(year)
        check_waste_daily(update, context)
    except (IndexError, ValueError) as e:
        update.message.reply_text("Some error occurred" + str(e))


def check_waste(update: Update, context: CallbackContext) -> None:
    chat_id = update.message.chat_id
    c = {"chat_id": chat_id, "update": update}
    waste_collection(c)


def check_waste_daily(update: Update, context: CallbackContext) -> None:
    """Add a job to the queue."""
    chat_id = update.message.chat_id
    try:
        due = time(20, 0, 0)

        c = {"chat_id": chat_id, "update": update}

        context.job_queue.run_daily(waste_collection, due, context=c, name=str(chat_id))

        # DEBUG
        # when = timedelta(seconds=5)
        # context.job_queue.run_once(waste_collection, when, context=c, name=str(chat_id))

        text = f'Successfully set! You will be warned at {due.strftime("%H:%M:%S")} daily. Use /stop to quit'
        update.message.reply_text(text)

    except (IndexError, ValueError) as e:
        update.message.reply_text("Some error occurred" + str(e))


def unset(update: Update, context: CallbackContext) -> None:
    """Remove the job if the user changed their mind."""
    chat_id = update.message.chat_id
    job_removed = remove_job_if_exists(str(chat_id), context)
    text = (
        "Timer successfully cancelled!" if job_removed else "You have no active timer."
    )
    update.message.reply_text(text)


def main() -> None:
    """Run bot."""
    # Create the Updater and pass it your bot's token.
    updater = Updater(TELEGRAM_SECRET_TOKEN)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("stop", unset))
    dispatcher.add_handler(CommandHandler("set", waste_set))
    dispatcher.add_handler(CommandHandler("today", check_waste))
    dispatcher.add_handler(CommandHandler("help", start))

    # Start the Bot
    updater.start_polling()

    # Block until you press Ctrl-C or the process receives SIGINT, SIGTERM or
    # SIGABRT. This should be used most of the time, since start_polling() is
    # non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == "__main__":
    main()
