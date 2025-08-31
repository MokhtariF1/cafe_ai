from telethon import TelegramClient, events
import requests
import uuid

# ----- تنظیمات -----
api_id = 86576
api_hash = "385886b58b21b7f3762e1cde2d651925"
bot_token = "8388745676:AAGziEb9JRqPAqd7JRJp0LXB6nWzxKdyN7M"

# نگهداری chat_id برای هر کاربر
user_sessions = {}


# ----- راه‌اندازی کلاینت -----
bot = TelegramClient('simple_bot', api_id, api_hash, proxy=("socks5", "127.0.0.1", 10808)).start(bot_token=bot_token)

# ----- هندل پیام -----
@bot.on(events.NewMessage(incoming=True))
async def handler(event):
    user_text = event.text
    user_id = event.sender_id

    # اگه کاربر چت آیدی نداره، بساز
    if user_id not in user_sessions:
        user_sessions[user_id] = str(uuid.uuid4())

    # دستور start
    if user_text == "/start":
        await event.reply("خوش اومدی! هر سوالی داری بپرس 🤖")
        return

    # دستور reset → ساخت chat_id جدید
    if user_text == "/reset":
        user_sessions[user_id] = str(uuid.uuid4())
        await event.reply("✅ کانتکست ریست شد. می‌تونی دوباره شروع کنی.")
        return

    chat_id = user_sessions[user_id]
    API_URL = f"http://127.0.0.1:8000/{chat_id}/chat"

    try:
        r = await event.reply("اوووم... بذار فکر کنم 🤔")
        resp = requests.post(API_URL, params={"query": user_text})

        if resp.status_code == 200:
            reply_text = str(resp.text)
            print(reply_text)
            reply_text = reply_text.replace("\\n", "\n")
            print(reply_text.count("\n"))
            print(reply_text.count("/n"))
            print(reply_text.count("n\\"))
            print(reply_text.count("\\n"))
        else:
            reply_text = f"❌ خطا از سمت سرور: {resp.status_code}"

    except Exception as e:
        reply_text = f"⚠️ خطا: {str(e)}"

    await event.reply(reply_text)
    print(user_sessions)
# ----- اجرا -----
print("🤖 Bot is running...")
bot.run_until_disconnected()



