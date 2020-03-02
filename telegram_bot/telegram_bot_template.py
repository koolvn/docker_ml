import telebot
import logging
import ssl
from telebot.types import Message
from aiohttp import web


BOT_TOKEN = '702667859:AAE2x5kNJvDe2FpI6V69GcIqWrX-L4v57PU'
OWNER_CHAT_ID = '208470137'
# Telegram WebHooks
WEBHOOK_HOST = '35.195.204.212'
WEBHOOK_PORT = 8443  # 443, 80, 88 or 8443 (port need to be 'open')
WEBHOOK_LISTEN = '0.0.0.0'  # In some VPS you may need to put here the IP addr

WEBHOOK_SSL_CERT = '/app/certificates/webhook_cert.pem'  # Path to the ssl certificate
WEBHOOK_SSL_PRIV = '/app/certificates/webhook_pkey.pem'  # Path to the ssl private key

# Quick'n'dirty SSL certificate generation:
#
# openssl genrsa -out webhook_pkey.pem 2048
# openssl req -new -x509 -days 3650 -key webhook_pkey.pem -out webhook_cert.pem
#
# When asked for "Common Name (e.g. server FQDN or YOUR name)" you should reply
# with the same value in you put in WEBHOOK_HOST

WEBHOOK_URL_BASE = "https://{}:{}".format(WEBHOOK_HOST, WEBHOOK_PORT)
WEBHOOK_URL_PATH = "/{}/".format(BOT_TOKEN)

logger = telebot.logger
telebot.logger.setLevel(logging.INFO)
bot = telebot.TeleBot(BOT_TOKEN)
app = web.Application()


# Process webhook calls
async def handle(request):
    if request.match_info.get('token') == bot.token:
        request_body_dict = await request.json()
        update = telebot.types.Update.de_json(request_body_dict)
        bot.process_new_updates([update])
        return web.Response()
    else:
        return web.Response(status=403)


app.router.add_post('/{token}/', handle)


# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Bot logic

# --------------------------------------------------------------------------------------------------------------------
# Keyboards
def default_keyboard(message):
    markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.row('/help')
    # if message.chat.id not in get_users_dict('telegram_users').keys():
    #     markup.row(types.KeyboardButton(text='/sign_up - Зарегистрироваться'))
    # else:
    #     markup.row(types.KeyboardButton(text='/settings - Настройки'))

    if message.chat.id == OWNER_CHAT_ID:
        markup.row(telebot.types.KeyboardButton('bot_log'))
    return markup


def remove_keyboard():
    keyboard = telebot.types.ReplyKeyboardRemove()
    return keyboard


# --------------------------------------------------------------------------------------------------------------------
# SERVICE DEFS

# --------------------------------------------------------------------------------------------------------------------
# HANDLERS


# Commands handlers
@bot.message_handler(commands=['start'])
def start_bot(message: Message):
    # print(message.json['from'])
    bot.send_message(message.chat.id, f'༼ つ ◕_◕ ༽つ'
                                      f'\nПривет, {message.chat.first_name}!',
                     reply_markup=default_keyboard(message))


@bot.message_handler(func=lambda message: message.text == 'bot_log')
def bot_log(message):
    if message.chat.id == OWNER_CHAT_ID:
        bot.send_document(OWNER_CHAT_ID, data=open('./bot_log.txt', 'rb'))


@bot.message_handler(content_types=['text'])
def echo_all(message: Message):
    bot.reply_to(message, '༼ つ ◕_◕ ༽つ\n' + message.text.upper())


# --------------------------------------------------------------------------------------------------------------------
# End of Bot logic
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# Remove webhook, it fails sometimes the set if there is a previous webhook
bot.remove_webhook()

# Set webhook
bot.set_webhook(url=WEBHOOK_URL_BASE + WEBHOOK_URL_PATH,
                certificate=open(WEBHOOK_SSL_CERT, 'r'))

# Build ssl context
context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
context.load_cert_chain(WEBHOOK_SSL_CERT, WEBHOOK_SSL_PRIV)

# Start web-server (aiohttp)
web.run_app(
    app,
    host=WEBHOOK_LISTEN,
    port=WEBHOOK_PORT,
    ssl_context=context,
)
