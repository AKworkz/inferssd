#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise JSON decoding and company-name rules without weakening public-content checks."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("public_scan", Path(__file__).with_name("public-scan.py"))
scanner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scanner)


class JsonLineTests(unittest.TestCase):
    def check_findings(self, text, expected, filename="fixture.json"):
        with contextlib.redirect_stdout(io.StringIO()):
            result = scanner.inspect(scanner.added_line_text(filename, text), "fixture:1")
        self.assertEqual(result, expected)

    def test_escaped_whitespace_before_path_marker(self):
        for separator in ("\t", "\n", "\r"):
            with self.subTest(separator=repr(separator)):
                self.check_findings(json.dumps(separator + "@case/input"), 0)

    def test_literal_login_is_still_blocked(self):
        self.check_findings(json.dumps("t" + "@case/input"), 1)

    def test_login_after_escaped_whitespace_is_still_blocked(self):
        self.check_findings(json.dumps("\t" + "user" + "@host"), 1)

    def test_unicode_escaped_login_is_blocked(self):
        self.check_findings('"\\u0075ser\\u0040host"', 1)

    def test_unicode_escaped_private_path_is_blocked(self):
        self.check_findings('"\\u002fhome\\u002ffixture\\u002finput"', 1)

    def test_double_escaped_tab_is_not_whitespace(self):
        self.check_findings(json.dumps("\\t" + "@case/input"), 1)

    def test_invalid_escape_is_preserved(self):
        text = '"\\q user' + '@host"'
        self.assertEqual(scanner.added_line_text("fixture.json", text), text)
        self.check_findings(text, 1)

    def test_unterminated_literal_is_preserved(self):
        text = '"user' + '@host'
        self.assertEqual(scanner.added_line_text("fixture.json", text), text)
        self.check_findings(text, 1)

    def test_trailing_escape_is_preserved(self):
        text = '"user' + '@host\\'
        self.assertEqual(scanner.added_line_text("fixture.json", text), text)
        self.check_findings(text, 1)

    def test_non_json_is_not_decoded(self):
        text = json.dumps("\t" + "@case/input")
        self.assertEqual(scanner.added_line_text("fixture.txt", text), text)
        self.check_findings(text, 1, filename="fixture.txt")

    def test_non_string_fragments_are_scanned(self):
        self.check_findings('"safe": user' + '@host', 1)

    def test_multiple_literals_and_fragments_keep_order(self):
        text = '{"key": "tab\\tquote\\\"slash\\\\", "next": 3}'
        self.assertEqual(scanner.added_line_text("fixture.json", text),
                         '{"key": "tab\tquote"slash\\", "next": 3}')

    def test_invalid_literal_does_not_hide_later_literal(self):
        text = '["\\q", "user' + '@host"]'
        self.check_findings(text, 1)

    def test_empty_and_plain_fragments_are_unchanged(self):
        for text in ("", '""', "42,", "  ]", '"a": 1,'):
            with self.subTest(text=text):
                self.assertEqual(scanner.added_line_text("fixture.json", text), text)


class CompanyNameTests(unittest.TestCase):
    # Blocked fixtures are split so that this file does not match its own patterns.
    def findings(self, text, strict):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = scanner.inspect(text, "fixture:1", strict)
        return result, output.getvalue()

    def test_company_domain_and_names_are_allowed(self):
        for text in ("https://wici.ai", '<a href="https://wici.ai/">', "(https://wici.ai).",
                     "WiCi AI", "wici-ai/inferssd"):
            for strict in (False, True):
                with self.subTest(text=text, strict=strict):
                    self.assertEqual(self.findings(text, strict), (0, ""))

    def test_other_company_spellings_are_blocked(self):
        for text in ("WI" + "CI-InferSSD", "WI" + "CI InferSSD", "wi" + "ci.air", "wi" + "ci.ai.example"):
            with self.subTest(text=text):
                self.assertEqual(self.findings(text, True)[0], 1)
                result, output = self.findings(text, False)
                self.assertEqual(result, 0)
                self.assertIn("REPORT", output)


class PowerShellLineTests(unittest.TestCase):
    def findings(self, text):
        with contextlib.redirect_stdout(io.StringIO()):
            return scanner.inspect(scanner.added_line_text("fixture.ps1", text), "fixture:1")

    def test_generated_batch_newlines_are_not_remote_login_identifiers(self):
        text = '"@echo blocked' + '`r`n' + '@exit /b 127' + '`r`n"'
        self.assertEqual(self.findings(text), 0)

    def test_real_login_and_literal_escaped_backtick_are_still_blocked(self):
        self.assertEqual(self.findings('"user' + '@host"'), 1)
        self.assertEqual(self.findings('"``n' + '@host"'), 1)
        # Single-quoted PowerShell text does not interpret newline escapes.
        self.assertEqual(self.findings("'literal`n" + "@host'"), 1)

    def test_escape_normalization_exposes_an_actual_personal_path(self):
        self.assertEqual(self.findings('"`n' + '/home/' + 'private-user/file"'), 1)


class TaskNumberTests(unittest.TestCase):
    # Fixtures are split so that this file does not match its own patterns.
    def findings(self, text, strict):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = scanner.inspect(text, "fixture:1", strict)
        return result, output.getvalue()

    def test_task_numbers_are_internal_wording(self):
        for text in ("// " + "Ta" + "sk 132 probe", "# " + "TA" + "SK12", "// " + "ta" + "sk-7",
                     "/* " + "Ta" + "sk #4 */"):
            with self.subTest(text=text):
                self.assertEqual(self.findings(text, True)[0], 1)
                result, output = self.findings(text, False)
                self.assertEqual(result, 0)
                self.assertIn("REPORT", output)

    def test_task_words_without_numbers_are_allowed(self):
        for text in ("// the task queue drains first", "# one task per request", "taskId = 3"):
            for strict in (False, True):
                with self.subTest(text=text, strict=strict):
                    self.assertEqual(self.findings(text, strict), (0, ""))


if __name__ == "__main__":
    unittest.main()
