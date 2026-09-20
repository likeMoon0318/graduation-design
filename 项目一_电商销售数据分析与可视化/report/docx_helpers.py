# -*- coding: utf-8 -*-
# Word 报告排版公共组件：页面、字体、标题层级、表格、插图与题注
# =====================================================================
# 说明：报告由代码生成，便于在拿到学校模板后一键重排。

from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

CN_BODY = "宋体"
CN_HEAD = "黑体"
EN_FONT = "Times New Roman"
BORDER = "D9D9D9"
HEADER_FILL = "2E5C8A"
ROW_ALT = "F2F6FA"
TEXT_WIDTH_CM = 15.8


def set_run_font(run, cn=CN_BODY, en=EN_FONT, size=12, bold=False, color=None):
    """同时设置西文与中文字体（python-docx 需要单独指定 eastAsia）。"""
    run.font.name = en
    run.font.size = Pt(size)
    run.font.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), en)
    rfonts.set(qn("w:hAnsi"), en)
    rfonts.set(qn("w:eastAsia"), cn)
    return run


def setup_document(doc):
    """设置 A4 页面、默认正文样式与页边距。"""
    for section in doc.sections:
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
        section.top_margin = Cm(2.6)
        section.bottom_margin = Cm(2.4)
        section.left_margin = Cm(2.6)
        section.right_margin = Cm(2.6)

    normal = doc.styles["Normal"]
    normal.font.name = EN_FONT
    normal.font.size = Pt(12)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), CN_BODY)
    pf = normal.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = 1.45
    pf.space_after = Pt(6)

    for name, size in (("Heading 1", 16), ("Heading 2", 14), ("Heading 3", 12.5)):
        st = doc.styles[name]
        st.font.name = EN_FONT
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = RGBColor.from_string("000000")
        st.element.rPr.rFonts.set(qn("w:eastAsia"), CN_HEAD)
        st.paragraph_format.space_before = Pt(12 if name == "Heading 1" else 9)
        st.paragraph_format.space_after = Pt(6)
        st.paragraph_format.line_spacing = 1.3
    return doc


def para(doc, text, size=12, indent=True, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
         space_after=6, bold=False):
    """正文段落：默认首行缩进两个字符、两端对齐。"""
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    if indent:
        p.paragraph_format.first_line_indent = Pt(size * 2)
    set_run_font(p.add_run(text), size=size, bold=bold)
    return p


def heading(doc, text, level=1):
    h = doc.add_heading(level=level)
    set_run_font(h.add_run(text), cn=CN_HEAD, en=EN_FONT,
                 size={1: 16, 2: 14, 3: 12.5}[level], bold=True, color="000000")
    return h


def caption(doc, text, above=False):
    """图表题注：居中、小一号字。"""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8 if above else 4)
    p.paragraph_format.space_after = Pt(4 if above else 10)
    set_run_font(p.add_run(text), size=10.5)
    return p


def add_figure(doc, image_path, cap, width_cm=None):
    """插入图片并自动加题注（居中、宽度自适应正文栏宽）。"""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True      # 图片与题注保持在同一页
    p.add_run().add_picture(str(image_path), width=Cm(width_cm or 14.6))
    return caption(doc, cap)


def _set_cell_borders(cell, color=BORDER, sz=6):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(sz))
        el.set(qn("w:color"), color)
        borders.append(el)
    tc_pr.append(borders)


def _shade(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def _cell_padding(cell, top=60, bottom=60, left=100, right=100):
    tc_pr = cell._tc.get_or_add_tcPr()
    mar = OxmlElement("w:tcMar")
    for name, val in (("top", top), ("bottom", bottom), ("left", left), ("right", right)):
        el = OxmlElement(f"w:{name}")
        el.set(qn("w:w"), str(val))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tc_pr.append(mar)


def add_table(doc, headers, rows, widths=None, size=10, align_center_cols=None,
              header_fill=HEADER_FILL, zebra=True):
    """插入灰边框表格：深色表头 + 交替行底色 + 文本垂直居中。"""
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    align_center_cols = align_center_cols if align_center_cols is not None else []

    hdr = table.rows[0].cells
    for i, text in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(3)
        set_run_font(p.add_run(str(text)), cn=CN_HEAD, size=size, bold=True, color="FFFFFF")
        _shade(hdr[i], header_fill)

    for r, row in enumerate(rows):
        cells = table.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.alignment = (WD_ALIGN_PARAGRAPH.CENTER if i in align_center_cols
                           else WD_ALIGN_PARAGRAPH.LEFT)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.2
            set_run_font(p.add_run(str(value)), size=size)

    for r, row in enumerate(table.rows):
        for cell in row.cells:
            _set_cell_borders(cell)
            _cell_padding(cell)
            if zebra and r > 0 and r % 2 == 0:
                _shade(cell, ROW_ALT)
    if widths:
        for row in table.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    table.rows[0].cells[0].paragraphs[0].paragraph_format.space_before = Pt(4)
    return table


def add_toc(doc):
    """插入目录域（在 Word 中按 F9 或右键「更新域」生成页码）。"""
    p = doc.add_paragraph()
    run = p.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = 'TOC \\o "1-3" \\h \\z \\u'
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    placeholder = OxmlElement("w:t")
    placeholder.text = "目录将在 Word 中更新域后自动生成。"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    for el in (fld_begin, instr, fld_sep, placeholder, fld_end):
        run._element.append(el)
    return p


def add_page_number_footer(doc, prefix=""):
    """在页脚居中插入页码域。"""
    for section in doc.sections:
        p = section.footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if prefix:
            set_run_font(p.add_run(prefix), size=9, color="6B7A8C")
        run = p.add_run()
        fld_begin = OxmlElement("w:fldChar")
        fld_begin.set(qn("w:fldCharType"), "begin")
        instr = OxmlElement("w:instrText")
        instr.set(qn("xml:space"), "preserve")
        instr.text = "PAGE"
        fld_end = OxmlElement("w:fldChar")
        fld_end.set(qn("w:fldCharType"), "end")
        for el in (fld_begin, instr, fld_end):
            run._element.append(el)
        set_run_font(run, size=9, color="6B7A8C")


def set_update_fields_on_open(doc):
    """让 Word 打开文档时自动更新域（目录页码、页码等）。"""
    settings = doc.settings.element
    update = settings.find(qn("w:updateFields"))
    if update is None:
        update = OxmlElement("w:updateFields")
        settings.append(update)
    update.set(qn("w:val"), "true")


def page_break(doc):
    from docx.enum.text import WD_BREAK
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def blank_line(doc, pt=6):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(pt)
    return p
