import os
import discord
from predict import Predict  # replace prediction_file with your ONNX file name

TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)

predictor = Prediction()

@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")

@bot.event
async def on_message(message):
    # Ignore bot's own messages
    if message.author == bot.user:
        return

    # Only react to Pokétwo messages
    if str(message.author) == "Pokétwo#8236":
        if message.attachments:
            for attachment in message.attachments:
                if attachment.url.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    await message.channel.send("🔍 Identifying Pokémon...")
                    try:
                        name, confidence = predictor.predict(attachment.url)
                        await message.channel.send(
                            f"🎯 I think it's **{name}** ({confidence} confident)"
                        )
                    except Exception as e:
                        await message.channel.send(f"❌ Error: {e}")

bot.run(TOKEN)
