# coding=utf-8
# Copyright 2023 The Fairseq Authors, Microsoft Research, and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Number Normalizer class for SpeechT5."""

import re


class EnglishNumberNormalizer:
    def __init__(self):
        self.ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        self.teens = [
            "",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ]
        self.tens = ["", "ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        self.thousands = [
            "",
            "thousand",
            "million",
            "billion",
            "trillion",
            "quadrillion",
            "quintillion",
            "sextillion",
            "septillion",
            "octillion",
            "nonillion",
            "decillion",
        ]

        # Define a dictionary to map currency symbols to their names
        # Top most traded currencies according to
        # https://en.wikipedia.org/wiki/Template:Most_traded_currencies
        self.currency_symbols = {
            "$": " dollars",
            "€": " euros",
            "£": " pounds",
            "¢": " cents",
            "¥": " japanese yen",
            "﷼": " saudi riyal",
            "₹": " indian rupees",
            "₽": " russian rubles",
            "฿": " thai baht",
            "₺": " turkish liras",
            "₴": " ukrainian hryvnia",
            "₣": " swiss francs",
            "₡": " costa rican colon",
            "₱": " philippine peso",
            "₪": " israeli shekels",
            "₮": " mongolian tögrög",
            "₩": " south korean won",
            "₦": " nigerian naira",
            "₫": " vietnamese Đồng",
        }

    def spell_number(self, num):
        if num == 0:
            return "zero"

        parts = []
        for i in range(0, len(self.thousands)):
            if num % 1000 != 0:
                part = ""
                hundreds = num % 1000 // 100
                tens_units = num % 100

                if hundreds > 0:
                    part += self.ones[hundreds] + " hundred"
                    if tens_units > 0:
                        part += " and "

                if tens_units > 10 and tens_units < 20:
                    part += self.teens[tens_units - 10]
                else:
                    tens_digit = self.tens[tens_units // 10]
                    ones_digit = self.ones[tens_units % 10]
                    if tens_digit:
                        part += tens_digit
                    if ones_digit:
                        if tens_digit:
                            part += " "
                        part += ones_digit

                parts.append(part)

            num //= 1000

        return " ".join(reversed(parts))

    def convert(self, number):
        """
        Converts an individual number passed in string form to spelt-out form
        """
        if "." in number:
            integer_part, decimal_part = number.split(".")
        else:
            integer_part, decimal_part = number, "00"

        # Extract currency symbol if present
        currency_symbol = ""
        for symbol, name in self.currency_symbols.items():
            if integer_part.startswith(symbol):
                currency_symbol = name
                integer_part = integer_part[len(symbol) :]
                break

            if integer_part.startswith("-"):
                if integer_part[1:].startswith(symbol):
                    currency_symbol = name
                    integer_part = "-" + integer_part[len(symbol) + 1 :]
                    break

        # Extract 'minus' prefix for negative numbers
        minus_prefix = ""
        if integer_part.startswith("-"):
            minus_prefix = "minus "
            integer_part = integer_part[1:]
        elif integer_part.startswith("minus"):
            minus_prefix = "minus "
            integer_part = integer_part[len("minus") :]

        percent_suffix = ""
        if "%" in integer_part or "%" in decimal_part:
            percent_suffix = " percent"
            integer_part = integer_part.replace("%", "")
            decimal_part = decimal_part.replace("%", "")

        integer_part = integer_part.zfill(3 * ((len(integer_part) - 1) // 3 + 1))

        parts = []
        for i in range(0, len(integer_part), 3):
            chunk = int(integer_part[i : i + 3])
            if chunk > 0:
                part = self.spell_number(chunk)
                unit = self.thousands[len(integer_part[i:]) // 3 - 1]
                if unit:
                    part += " " + unit
                parts.append(part)

        spelled_integer = " ".join(parts)

        # Format the spelt-out number based on conditions, such as:
        # If it has decimal parts, currency symbol, minus prefix, etc
        if decimal_part == "00":
            return (
                f"{minus_prefix}{spelled_integer}{percent_suffix}{currency_symbol}"
                if minus_prefix or currency_symbol
                else f"{spelled_integer}{percent_suffix}"
            )
        else:
            spelled_decimal = " ".join([self.spell_number(int(digit)) for digit in decimal_part])
            return (
                f"{minus_prefix}{spelled_integer} point {spelled_decimal}{percent_suffix}{currency_symbol}"
                if minus_prefix or currency_symbol
                else f"{minus_prefix}{spelled_integer} point {spelled_decimal}{percent_suffix}"
            )

    def __call__(self, text):
        """
        Convert numbers / number-like quantities in a string to their spelt-out counterparts
        """
        # Form part of the pattern for all currency symbols
        pattern = r"(?<!\w)(-?\$?\€?\£?\¢?\¥?\₹?\₽?\฿?\₺?\₴?\₣?\₡?\₱?\₪?\₮?\₩?\₦?\₫?\﷼?\d+(?:\.\d{1,2})?%?)(?!\w)"

        # Find and replace commas in numbers (15,000 -> 15000, etc)
        text = re.sub(r"(\d+,\d+)", lambda match: match.group(1).replace(",", ""), text)

        # Use regex to find and replace numbers in the text
        converted_text = re.sub(pattern, lambda match: self.convert(match.group(1)), text)
        converted_text = re.sub(" +", " ", converted_text)

        return converted_text
