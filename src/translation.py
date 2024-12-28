import logging

from langdetect import detect
import os
os.environ["ARGOS_PACKAGES_DIR"] = r"..\language\argos-translate"

import argostranslate.package
import argostranslate.translate
import warnings
from app_logger import AppLogger


# Suppress FutureWarning messages
warnings.filterwarnings("ignore", category=FutureWarning)


class TranslationException(Exception):
    """Custom Exception for Translation Failures"""

    def __init__(self, message):
        super().__init__(message)
        self.message = message


class TranslationService:
    # Map of alternate languages or replacements for unsupported language codes
    LANGUAGE_CODE_MAP = {
        "zh-cn": "zh",  # Map Simplified Chinese to Chinese
    }

    def __init__(self):
        self.logger = AppLogger(file_name="Tran.log", overwrite=True, log_level=logging.DEBUG)
        self.logger.debug("[TRANSLATION_SERVICE] Initializing TranslationService...")

    def map_language_code(self, lang_code):
        """Map detected language code to a supported Argos Translate language code."""
        return self.LANGUAGE_CODE_MAP.get(lang_code, lang_code)

    def auto_translate(self, text, target_lang_code):
        """
        Translates text into the specified target language with auto-installation of missing models,
        incorporating normalized language codes.
        """
        try:
            # if text is None:
            #     return ""
            target_lang_code = self.map_language_code(target_lang_code)
            detected_language = detect(text)
            mapped_source_lang = self.map_language_code(detected_language)
            self.logger.info(f"Detected source language: {mapped_source_lang} (mapped)")
            # if mapped_source_lang == target_lang_code:
            #     self.logger.debug(f"Source language and target language are the same ({mapped_source_lang}). Returning original text.")
            #     return text
            while True:
                try:
                    self.logger.debug(f"Attempting direct translation {mapped_source_lang} → {target_lang_code}...")
                    return self.translate_direct(text, mapped_source_lang, target_lang_code)
                except TranslationException as e:
                    if "not supported" in e.message or "not installed" in e.message:
                        self.logger.debug(f"Missing model for {mapped_source_lang} → {target_lang_code}. Attempting to download...")
                        self._install_language_pair(mapped_source_lang, target_lang_code)
                    elif "No direct translation model" in e.message:
                        self.logger.debug(f"No direct model for {mapped_source_lang} → {target_lang_code}. Falling back to intermediate...")
                        return self._translate_with_intermediate(text, mapped_source_lang, target_lang_code)  # Adjust intermediate as necessary
                    else:
                        raise e
        except TranslationException as e:
            return f"Translation Error: {e.message}"

    def translate_direct(self, text, source_lang_code, target_lang_code):
        """
        Translate directly between two installed languages.
        """
        if text is None:
            return ""
        if source_lang_code == target_lang_code:
            return text
        from_lang = self._get_installed_language(source_lang_code)
        to_lang = self._get_installed_language(target_lang_code)
        if not from_lang:
            raise TranslationException(f"Source language '{source_lang_code}' is not supported or installed.")
        if not to_lang:
            raise TranslationException(f"Target language '{target_lang_code}' is not supported or installed.")
        translation = from_lang.get_translation(to_lang)
        if not translation:
            raise TranslationException(f"No direct translation model available for {source_lang_code} → {target_lang_code}")
        self.logger.debug(f"Translating from {from_lang.name} to {to_lang.name}...")
        return translation.translate(text)

    def _translate_with_intermediate(self, text, intermediate_lang_code, target_lang_code):
        """
        Translates a text with a fallback intermediate pathway, with normalized language codes.
        """
        if text is None:
            return ""
        if intermediate_lang_code == target_lang_code:
            return text
        intermediate_lang_code = self.map_language_code(intermediate_lang_code)
        target_lang_code = self.map_language_code(target_lang_code)
        detected_language = detect(text)
        mapped_source_lang = self.map_language_code(detected_language)
        self.logger.debug(f"Detected source language: {mapped_source_lang}")
        self.logger.debug(f"Ensuring models for {mapped_source_lang} → {intermediate_lang_code} and {intermediate_lang_code} → {target_lang_code}...")
        self._install_language_pair(mapped_source_lang, intermediate_lang_code)  # Step 1
        self._install_language_pair(intermediate_lang_code, target_lang_code)  # Step 2
        self.logger.debug(f"Translating from {mapped_source_lang} to {intermediate_lang_code} (Step 1)...")
        intermediate_translation = self.translate_direct(text, mapped_source_lang, intermediate_lang_code)
        self.logger.debug(f"Translating from {intermediate_lang_code} to {target_lang_code} (Step 2)...")
        return self.translate_direct(intermediate_translation, intermediate_lang_code, target_lang_code)

    @staticmethod
    def _get_installed_language(lang_code):
        """Helper to fetch installed languages."""
        installed_languages = argostranslate.translate.get_installed_languages()
        return next((lang for lang in installed_languages if lang.code == lang_code), None)

    def _install_language_pair(self, from_lang_code, to_lang_code):
        """Ensure the required language pair is installed, normalizing language codes."""
        from_lang_code = self.map_language_code(from_lang_code)
        to_lang_code = self.map_language_code(to_lang_code)
        if from_lang_code == to_lang_code:
            self.logger.debug(f"Skipping installation for {from_lang_code} → {to_lang_code}: invalid pair (identical).")
            return
        self.logger.debug(f"Ensuring model for: {from_lang_code} → {to_lang_code}")
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        package = next((p for p in available_packages if p.from_code == from_lang_code and p.to_code == to_lang_code), None)
        if package:
            self.logger.debug(f"Found package for: {from_lang_code} → {to_lang_code}. Installing...")
            package.install()
            self.logger.debug(f"Model installed successfully: {from_lang_code} → {to_lang_code}")
        else:
            raise TranslationException(f"No package available for: {from_lang_code} → {to_lang_code}")

    @staticmethod
    def list_installed_languages():
        """List all installed languages and their codes."""
        installed_languages = argostranslate.translate.get_installed_languages()
        for language in installed_languages:
            print(f"{language.name} ({language.code})")

if __name__ == "__main__":

    translation_service = TranslationService()

    # Example input
    source_text = "We made a very thorough assessment"

    translated_text = translation_service.auto_translate(source_text, "fr")
    print("Final Translation:", translated_text)
    translated_text = translation_service.auto_translate(source_text, "de")
    print("Final Translation:", translated_text)
    translated_text = translation_service.auto_translate(source_text, "zh")
    print("Final Translation:", translated_text)
    translated_text = translation_service.auto_translate(source_text, "es")
    print("Final Translation:", translated_text)
    translated_text = translation_service.auto_translate(source_text, "sv")
    print("Final Translation:", translated_text)

    source_text = "Nous avons fait une évaluation très approfondie"
    translated_text = translation_service.auto_translate(source_text, "en")
    print("Final Translation (en):", translated_text)
    source_text = "Wir haben eine sehr gründliche Bewertung vorgenommen"
    translated_text = translation_service.auto_translate(source_text, "en")
    print("Final Translation (en):", translated_text)
    source_text = "我们做了非常彻底的评估"
    translated_text = translation_service.auto_translate(source_text, "en")
    print("Final Translation (en):", translated_text)
    source_text = "Hicimos una evaluación muy exhaustiva"
    translated_text = translation_service.auto_translate(source_text, "en")
    print("Final Translation (en):", translated_text)
    source_text = "Vi gjorde en mycket grundlig bedömning"
    translated_text = translation_service.auto_translate(source_text, "en")
    print("Final Translation (en):", translated_text)
