"""
Unit tests for Scripts/ProcessHTMLs/preprocessHTMLs.py.

Covers:
  - remove_metadata_lines: pure function that removes metadata lines from the body.
  - strip_unwanted_elements: removes unwanted HTML elements from a BeautifulSoup object.
"""

import pytest
from bs4 import BeautifulSoup

from Scripts.ProcessHTMLs.preprocessHTMLs import remove_metadata_lines, strip_unwanted_elements


# Helpers
def make_soup(html: str) -> BeautifulSoup:
    # Converts an HTML string into a BeautifulSoup object for use in tests.
    return BeautifulSoup(html, "html.parser")


# remove_metadata_lines
class TestRemoveMetadataLines:
    def test_removes_source_line(self):
        # The line that exactly matches `source` must be removed from the body.
        body = "Congreso de la República\nARTÍCULO 1. Contenido real."
        result = remove_metadata_lines(body, source="Congreso de la República", subtipo="")
        assert "Congreso de la República" not in result
        assert "ARTÍCULO 1. Contenido real." in result

    def test_removes_subtipo_label_and_value(self):
        # The 'subtipo:' label and its value on the next line must both be removed.
        body = "Introducción del documento\nsubtipo:\nLey\nContenido del artículo."
        result = remove_metadata_lines(body, source="", subtipo="Ley")
        assert "subtipo:" not in result.lower()
        assert "Ley" not in result
        assert "Contenido del artículo." in result

    def test_preserves_lines_that_dont_match(self):
        # Lines that do not match source or subtipo must be kept intact.
        body = "Primera línea\nSegunda línea\nTercera línea"
        result = remove_metadata_lines(body, source="Otra fuente", subtipo="Decreto")
        assert result == body

    def test_empty_source_and_subtipo_returns_body_unchanged(self):
        # With empty source and subtipo, no lines should be removed.
        body = "Texto legal sin metadatos especiales."
        result = remove_metadata_lines(body, source="", subtipo="")
        assert result == body

    def test_empty_body_returns_empty(self):
        # An empty body must return an empty string without raising an exception.
        result = remove_metadata_lines("", source="Fuente", subtipo="Ley")
        assert result == ""

    def test_removes_blank_lines_after_source(self):
        # Blank lines that follow the source line must also be skipped.
        body = "Congreso de la República\n\n\nARTÍCULO 1. Norma."
        result = remove_metadata_lines(body, source="Congreso de la República", subtipo="")
        all_lines = result.splitlines()
        lines = [line for line in all_lines if line.strip() != ""]
        assert all("Congreso" not in line for line in lines)


# strip_unwanted_elements
class TestStripUnwantedElements:
    def test_removes_script_tags(self):
        # <script> tags must be removed to avoid executable code in extracted text.
        soup = make_soup("<div><script>alert('xss')</script><p>Contenido legal.</p></div>")
        strip_unwanted_elements(soup)
        assert soup.find("script") is None
        assert soup.find("p").get_text() == "Contenido legal."

    def test_removes_style_tags(self):
        # <style> tags contain CSS, not legal content, and must be removed.
        soup = make_soup("<div><style>body{color:red}</style><p>Texto.</p></div>")
        strip_unwanted_elements(soup)
        assert soup.find("style") is None

    def test_removes_nav_and_footer(self):
        # Navigation and footer elements are UI components, not legal content.
        soup = make_soup("<nav>Menú</nav><main>Artículo</main><footer>Pie</footer>")
        strip_unwanted_elements(soup)
        assert soup.find("nav") is None
        assert soup.find("footer") is None
        assert soup.find("main") is not None

    def test_removes_hidden_display_none_elements(self):
        # Elements with display:none are hidden on the page and not useful content.
        soup = make_soup('<div style="display:none">Oculto</div><p>Visible</p>')
        strip_unwanted_elements(soup)
        assert "Oculto" not in soup.get_text()
        assert "Visible" in soup.get_text()

    def test_removes_hidden_visibility_hidden_elements(self):
        # Elements with visibility:hidden must also be excluded from extracted text.
        soup = make_soup('<span style="visibility:hidden">Invisible</span><p>Real</p>')
        strip_unwanted_elements(soup)
        assert "Invisible" not in soup.get_text()

    def test_removes_toc_div_by_id(self):
        # A div with id="toc" is the table of contents, which is navigation UI.
        soup = make_soup('<div id="toc">Tabla de contenidos</div><p>Norma.</p>')
        strip_unwanted_elements(soup)
        assert soup.find("div", id="toc") is None
        assert "Norma." in soup.get_text()

    def test_removes_toc_toggle_by_class(self):
        # Elements with class "toctoggle" are UI buttons from the SUIN portal.
        soup = make_soup('<span class="toctoggle">[Mostrar]</span><p>Contenido.</p>')
        strip_unwanted_elements(soup)
        assert soup.find("span", class_="toctoggle") is None

    def test_preserves_content_elements(self):
        # Real content elements like <article> and <p> must not be removed.
        soup = make_soup("<article><p>Artículo 1. Norma general.</p></article>")
        strip_unwanted_elements(soup)
        assert "Artículo 1. Norma general." in soup.get_text()

    def test_empty_html_does_not_raise(self):
        # An empty HTML string must not raise any exception.
        soup = make_soup("")
        strip_unwanted_elements(soup)
        assert soup.get_text() == ""
