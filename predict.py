import os
import discord
import json
import random
from predict import Prediction  # your predictor

TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.all()
bot = discord.Client(intents=intents)

predictor = Prediction()

# Load Pokémon data file
with open("pokemon_data.txt", "r", encoding="utf-8") as f:
    pokemon_data = json.load(f)

@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")

def get_pokemon_info(label_name):
    # Extract dex number (first 4 digits from label)
    dex_number = label_name[:4].lstrip("0")  # remove leading zeros
    if dex_number == "":
        dex_number = "0"
    # Search JSON
    for key, poke in pokemon_data.items():
        if str(poke.get("dex_number")) == dex_number:
            # Choose a random name from "names" dict
            names_dict = poke.get("names", {})
            random_name = None
            if names_dict:
                random_name = random.choice(list(names_dict.values()))
            return poke.get("name"), random_name, poke.get("dex_number")
    return label_name, None, None

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Manual predict
    if message.content.startswith("!identify "):
        url = message.content.split(" ", 1)[1]
        await message.channel.send("🔍 Identifying Pokémon...")
        try:
            label_name, confidence = predictor.predict(url)
            name, random_name, dex = get_pokemon_info(label_name)
            reply = f"🎯 Predicted Pokémon: **{name}** (confidence: {confidence})\n"
            if random_name:
                reply += f"🌐 Random Name: {random_name}\n"
            if dex:
                reply += f"📖 Dex Number: {dex}"
            await message.channel.send(reply)
        except Exception as e:
            await message.channel.send(f"❌ Error: {e}")

    # Auto detect Pokétwo spawns
    if message.author.id == 716390085896962058:
        image_url = None
        if message.attachments:
            for attachment in message.attachments:
                if attachment.url.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    image_url = attachment.url
        if not image_url and message.embeds:
            embed = message.embeds[0]
            if embed.image and embed.image.url:
                image_url = embed.image.url

        if image_url:
            await message.channel.send("🔍 Identifying Pokémon...")
            try:
                label_name, confidence = predictor.predict(image_url)
                name, random_name, dex = get_pokemon_info(label_name)
                reply = f"🎯 Predicted Pokémon: **{name}** (confidence: {confidence})\n"
                if random_name:
                    reply += f"🌐 Random Name: {random_name}\n"
                if dex:
                    reply += f"📖 Dex Number: {dex}"
                await message.channel.send(reply)
            except Exception as e:
                await message.channel.send(f"❌ Error: {e}")

bot.run(TOKEN)
