import re

from lm_eval.api.filter import Filter


class NormalizeBracedThousands(Filter):
    """Normalize complete braced thousands groups before flexible extraction.

    Scan the whole numeric token first: searching for valid groups directly could
    normalize a suffix of an invalid token such as ``1234{,}567{,}890``.
    Signs, currency prefixes, and surrounding LaTeX are left untouched.
    """

    _numeric_token = re.compile(r"(?:[0-9.,]|\{,\})+")
    _grouped_number = re.compile(r"[0-9]{1,3}(?:(?:,|\{,\})[0-9]{3})+(?:\.[0-9]+)?")

    def apply(self, resps, docs):
        def normalize(match):
            token = match.group()
            # Keep sentence-ending periods for the existing extraction/scoring
            # rules, including their treatment of repeated punctuation.
            if "{,}" in token and self._grouped_number.fullmatch(token.rstrip(".")):
                return token.replace("{,}", ",")
            return token

        return [
            [
                self._numeric_token.sub(normalize, response)
                if isinstance(response, str)
                else response
                for response in responses
            ]
            for responses in resps
        ]
