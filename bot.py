import os
import discord
from discord.ext import commands
from main_tensor import PokeNet  # import your classifier

TOKEN = os.getenv("DISCORD_TOKEN")  # set this in Railway Variables

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Load the classifier
net = PokeNet()

@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")

@bot.command()
async def identify(ctx, url: str):
    """Identify Pokémon from an image URL"""
    await ctx.send("🔍 Analyzing image...")
    name, acc = net.predict_url(url)
    if name:
        await ctx.send(f"🎯 I think this is **{name}** ({acc}% confidence)")
    else:
        await ctx.send("❌ Sorry, I couldn't identify the Pokémon.")

bot.run(TOKEN)
