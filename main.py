from prompts.general_prompt import general_chain
from prompts.classifier_prompt import classifier_chain

import asyncio
import os

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from prompts.history_pipeline import history_pipeline
from config import TG_TOKEN


bot = Bot(
    token=TG_TOKEN,
    default=DefaultBotProperties(
        parse_mode=ParseMode.HTML
    ),
)

dp = Dispatcher()


@dp.message(F.text)
async def message_handler(message):
    try:
        result = await asyncio.to_thread(
            classifier_chain.invoke,
            {"query": message.text}
        )
        print(result)
        if result["answer"].lower() == "загальний":
            result = await asyncio.to_thread(
                        general_chain.invoke,
                        {"query": message.text},
                    )

            print(result)
            if len(result) <= 4096:
                await message.answer(result)
            else:
                for start in range(0, len(result), 4096):
                    await message.answer(result[start:start + 4096])

        elif result["answer"].lower() == "історичний":
            result = await asyncio.to_thread(
                    history_pipeline.invoke,
                    {"query": message.text},
                )

            response = result["checked_answer"]
            if len(response) <= 4096:
                await message.answer(response)
            else:
                for start in range(0, len(response), 4096):
                    await message.answer(response[start:start + 4096])


    except Exception:
        await message.answer(
            "Сталася помилка під час обробки запиту. Спробуй ще раз."
        )
        raise


async def main():
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
