from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
import tensorflow_hub as hub
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np


api = "5900936236:AAFA641RONavuzZFy4EdY6Y7Q_VOfnPxMrw"


async def start(update: Update, context):
    if update.message:
        first_name = update.effective_user.first_name
        last_name = update.effective_user.last_name
        await update.message.reply_text(f"Welcome {first_name} {last_name}")
        await update.message.reply_text("Send your photo")
    else:
        return


async def rmg(update: Update, context):
    if update.message.photo:
        photo = await update.message.photo[-1].get_file()
        await photo.download_to_drive("image.jpg")
        await update.message.reply_text("Select style")
        await update.message.reply_text(
            "~PAINTINGS\n"
            "1) The Mona Lisa - 1\n"
            "2) The Starry Night - 2\n"
            "3) Picasso Art - 3\n"
            "4) Oil Painting - 4\n"
            "5) Oil Painting (2) - 5\n"
            "6) Oil Painting (3) - 6\n"
            "7) Oil Painting (4) - 7"
        )
        await update.message.reply_text("Send your command (Enter a number between 1 and 7)")
    else:
        await update.message.reply_text("Please upload a valid photo.")

async def painting(update: Update, context):
    try:

        st = ["The Mona Lisa", "The Starry Night", "Picasso Art", "Oil Painting 1", 
              "Oil Painting 2", "Oil Painting 3", "Oil Painting 4"]
        styles = [
            "monolisa.webp", "stary night.webp", "picaso.jpg", "oil-painting-1.webp",
            "oil-painting-2.webp", "oil-painting-3.webp", "anmie1.jpg"
        ]

        query = int(update.message.text)
        if query < 1 or query > 7:
            raise ValueError("Invalid selection")

        await update.message.reply_text(f"You've selected {st[query - 1]} art.")
        await update.message.reply_text("Wait for a few minutes, your image is processing...")

        content_image = plt.imread("image.jpg")
        style_image = plt.imread(styles[query - 1])

        content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.0
        style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.0
        style_image = tf.image.resize(style_image, (256, 256))

        hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
        outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
        stylized_image = outputs[0]

        saving_image = np.squeeze(stylized_image)
        plt.axis('off')
        plt.imsave("temp.jpg", saving_image)

        await update.message.reply_text("Process is done, thank you for waiting.")
        await update.message.reply_photo(photo=open('temp.jpg', 'rb'))

    except ValueError:
        await update.message.reply_text("Invalid command. Please enter a number between 1 and 7.")
    except Exception as e:
        await update.message.reply_text(f"An error occurred: {e}")

def main():

    application = Application.builder().token(api).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, rmg))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, painting))

    application.run_polling()

if __name__ == "__main__":

    main()