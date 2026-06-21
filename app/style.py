"""Application stylesheet for the PySide desktop UI."""

APP_STYLESHEET = """
QMainWindow {
    background: #ece7dc;
}

QWidget#AppSurface,
QScrollArea,
QScrollArea > QWidget > QWidget {
    background: #ece7dc;
}

QWidget#Hero {
    background: #1f2523;
    border: 1px solid #323a36;
    border-radius: 8px;
}

QLabel {
    color: #24231f;
    font-size: 13px;
}

QLabel#AppTitle {
    color: #fff8ec;
    font-size: 28px;
    font-weight: 800;
}

QLabel#AppSubtitle {
    color: #d6caba;
    font-size: 13px;
}

QLabel#HeaderPill {
    color: #10231f;
    background: #b9eadf;
    border-radius: 8px;
    padding: 7px 12px;
    font-weight: 700;
}

QWidget#ModeBar {
    background: transparent;
}

QWidget#FrontendPage {
    background: #07090d;
    border-radius: 8px;
}

QLabel#FrontendBackdrop {
    background: #07090d;
    border-radius: 8px;
}

QWidget#FrontendLoadingOverlay {
    background: rgba(4, 5, 12, 218);
    border-radius: 8px;
}

QLabel#FrontendLoadingTitle {
    color: #fff8ff;
    font-size: 38px;
    font-weight: 900;
}

QLabel#FrontendLoadingStatus {
    color: rgba(255, 248, 255, 215);
    font-size: 14px;
    font-weight: 800;
}

QProgressBar#FrontendLoadingProgress {
    background: rgba(255, 255, 255, 76);
    border: 1px solid rgba(255, 255, 255, 120);
    border-radius: 6px;
    height: 10px;
    max-width: 360px;
}

QProgressBar#FrontendLoadingProgress::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff4fd8, stop:0.52 #9d56ff, stop:1 #57e9d7);
    border-radius: 5px;
}

QWidget#FrontendContent {
    background: rgba(0, 0, 0, 78);
    border-radius: 8px;
}

QWidget#GameBox {
    background: rgba(9, 13, 18, 178);
    border: 1px solid rgba(255, 255, 255, 42);
    border-radius: 8px;
}

QLabel#FrontendControlLabel {
    color: #fff7ed;
    font-size: 13px;
    font-weight: 800;
}

QLabel#MutedLabel {
    color: #71695e;
}

QLabel#FilePath {
    color: #2e2d29;
    background: #f4efe4;
    border: 1px solid #ded4c2;
    border-radius: 6px;
    padding: 9px 11px;
}

QLabel#StatusLabel {
    color: #19352f;
    background: #dff4ee;
    border: 1px solid #afddd0;
    border-radius: 6px;
    padding: 9px 11px;
    font-weight: 600;
}

QLabel#PeopleLabel {
    color: #4b4037;
    background: #fff9ef;
    border: 1px solid #eadcc7;
    border-radius: 6px;
    padding: 8px 10px;
}

QLabel#PanelTitle {
    color: #292720;
    font-size: 15px;
    font-weight: 800;
}

QLabel#VideoPreview {
    background: #111313;
    color: #b9b6ad;
    border: 1px solid #252b29;
    border-radius: 8px;
    padding: 14px;
}

QLabel#FrontendStage {
    background: rgba(0, 0, 0, 96);
    color: rgba(255, 255, 255, 202);
    border: 1px solid rgba(255, 255, 255, 46);
    border-radius: 8px;
    padding: 14px;
    font-size: 18px;
    font-weight: 800;
}

QLabel#ScoreCard {
    color: #1f2421;
    background: #fff9ef;
    border: 1px solid #ead9bd;
    border-radius: 8px;
    padding: 12px;
    font-weight: 600;
}

QGroupBox {
    background: #fffdf8;
    border: 1px solid #d9cdbb;
    border-radius: 8px;
    margin-top: 18px;
    padding: 18px 14px 14px 14px;
    color: #292720;
    font-size: 15px;
    font-weight: 800;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 6px;
    background: #fffdf8;
}

QGroupBox#LibraryBox {
    background: #f8f3e9;
    border-color: #e5d8c4;
}

QGroupBox#LibraryBox::title {
    background: #f8f3e9;
}

QPushButton {
    background: #f4efe4;
    border: 1px solid #cfc2ad;
    border-radius: 6px;
    color: #292720;
    font-weight: 700;
    padding: 9px 12px;
}

QPushButton:hover {
    background: #ede2d1;
}

QPushButton:pressed {
    background: #dfd1bd;
}

QPushButton:disabled {
    color: #9a9285;
    background: #eee8de;
    border-color: #dfd7ca;
}

QPushButton[variant="primary"] {
    background: #137c70;
    border-color: #0d665b;
    color: #fffdf8;
}

QPushButton[variant="primary"]:hover {
    background: #0f6f64;
}

QPushButton[variant="accent"] {
    background: #c75f29;
    border-color: #ad4f20;
    color: #fffdf8;
}

QPushButton[variant="danger"] {
    background: #fff4ed;
    border-color: #dfb99d;
    color: #8f3414;
}

QPushButton[variant="compact"] {
    padding: 7px 10px;
}

QPushButton[modeToggle="true"] {
    background: #e6ddcf;
    border-color: #cbbca5;
    min-width: 108px;
}

QPushButton[modeToggle="true"]:checked,
QPushButton[active="true"] {
    background: #1f2523;
    border-color: #1f2523;
    color: #fff8ec;
}

QComboBox,
QTextEdit {
    background: #fffaf2;
    border: 1px solid #cfc2ad;
    border-radius: 6px;
    color: #24231f;
    padding: 7px 9px;
    selection-background-color: #137c70;
    selection-color: #fffdf8;
}

QComboBox:disabled,
QTextEdit:disabled {
    color: #9a9285;
    background: #eee8de;
}

QWidget#FrontendPage QComboBox {
    background: rgba(255, 255, 255, 226);
    border-color: rgba(255, 255, 255, 110);
    color: #14161a;
    min-height: 20px;
}

QWidget#FrontendPage QPushButton {
    background: rgba(255, 255, 255, 216);
    border-color: rgba(255, 255, 255, 112);
    color: #11151a;
}

QWidget#FrontendPage QPushButton:hover {
    background: rgba(255, 255, 255, 238);
}

QWidget#FrontendPage QPushButton[variant="primary"] {
    background: #00d6c9;
    border-color: #52fff3;
    color: #051013;
}

QWidget#FrontendPage QPushButton[variant="primary"]:hover {
    background: #44eee4;
}

QWidget#FrontendPage QLabel#StatusLabel {
    color: #f6fffb;
    background: rgba(11, 31, 36, 186);
    border-color: rgba(97, 255, 230, 92);
}

QWidget#FrontendPage {
    font-family: "Arial Rounded MT Bold", "Avenir Next", "Helvetica Neue";
}

QWidget#FrontendGameShell,
QWidget#DanceSelector,
QWidget#PlayPanel,
QWidget#DanceCardsHost {
    background: transparent;
}

QLabel#FrontendKicker {
    color: #70f5dd;
    font-size: 12px;
    font-weight: 900;
}

QLabel#FrontendTitle {
    color: #fff8ff;
    font-size: 40px;
    font-weight: 900;
}

QLabel#FrontendSongMeta {
    color: rgba(255, 247, 255, 210);
    font-size: 14px;
    font-weight: 800;
}

QLabel#FrontendReadyBadge {
    color: #ffffff;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ff4fd8, stop:0.52 #a85bff, stop:1 #63f0dc);
    border: 2px solid rgba(255, 255, 255, 180);
    border-radius: 8px;
    padding: 13px 22px;
    font-size: 26px;
    font-weight: 900;
}

QLabel#FrontendPanelTitle,
QLabel#FrontendSongTitle {
    color: #fff8ff;
    font-size: 20px;
    font-weight: 900;
}

QLabel#FrontendSongTitle {
    font-size: 24px;
}

QLabel#FrontendMiniPill {
    color: #4d258d;
    background: rgba(255, 255, 255, 226);
    border: 1px solid rgba(255, 255, 255, 170);
    border-radius: 8px;
    padding: 6px 10px;
    font-weight: 900;
}

QWidget#ArtistStrip {
    background: rgba(255, 255, 255, 74);
    border: 1px solid rgba(255, 255, 255, 88);
    border-radius: 8px;
}

QScrollArea#DanceCardScroll {
    background: transparent;
    border: 0;
}

QScrollArea#DanceCardScroll > QWidget > QWidget {
    background: transparent;
}

QLabel#FrontendEmptyState {
    color: rgba(255, 248, 255, 220);
    background: rgba(18, 12, 35, 128);
    border: 1px solid rgba(255, 255, 255, 80);
    border-radius: 8px;
    padding: 28px;
    font-size: 16px;
    font-weight: 800;
}

QPushButton#CompanyTab {
    color: #5d2fa6;
    background: rgba(255, 255, 255, 228);
    border: 2px solid rgba(255, 220, 255, 190);
    border-radius: 8px;
    padding: 10px 14px;
    min-width: 78px;
    font-size: 14px;
    font-weight: 900;
}

QPushButton#CompanyTab:hover {
    background: #fff9ff;
    border-color: #ff92db;
}

QPushButton#CompanyTab[selected="true"] {
    color: #ffffff;
    background: #a35cff;
    border-color: #ffd8fb;
}

QPushButton#CompanyTab[company="SM"][selected="true"] {
    background: #8a6cff;
}

QPushButton#CompanyTab[company="JYP"][selected="true"] {
    background: #13c7b4;
}

QPushButton#CompanyTab[company="YG"][selected="true"] {
    background: #242338;
}

QPushButton#CompanyTab[company="HYBE"][selected="true"] {
    background: #f458b5;
}

QPushButton#ArtistChip,
QPushButton#PlayerChip {
    color: #6a35ac;
    background: rgba(255, 255, 255, 226);
    border: 1px solid rgba(255, 210, 255, 180);
    border-radius: 8px;
    padding: 8px 11px;
    font-weight: 900;
}

QPushButton#ArtistChip[selected="true"],
QPushButton#PlayerChip[selected="true"] {
    color: #0b2430;
    background: #68ead8;
    border-color: #cffff8;
}

QPushButton#CompanyTab:disabled,
QPushButton#ArtistChip:disabled,
QPushButton#PlayerChip:disabled {
    color: rgba(70, 48, 88, 145);
    background: rgba(255, 255, 255, 126);
    border-color: rgba(255, 255, 255, 86);
}

QPushButton#DanceCard {
    color: #5f34ad;
    background: rgba(255, 255, 255, 236);
    border: 2px solid rgba(255, 255, 255, 160);
    border-radius: 8px;
    padding: 12px;
    font-size: 14px;
    font-weight: 900;
    text-align: left;
}

QPushButton#DanceCard:hover {
    background: #fff8ff;
    border-color: #ff9de3;
}

QPushButton#DanceCard[selected="true"] {
    color: #ffffff;
    background: #b765f2;
    border-color: #ffffff;
}

QPushButton#DanceCard:disabled {
    color: rgba(255, 255, 255, 132);
    background: rgba(255, 255, 255, 76);
    border-color: rgba(255, 255, 255, 58);
}

QPushButton#DanceCard[company="SM"] {
    border-left: 6px solid #8a6cff;
}

QPushButton#DanceCard[company="JYP"] {
    border-left: 6px solid #13c7b4;
}

QPushButton#DanceCard[company="YG"] {
    border-left: 6px solid #242338;
}

QPushButton#DanceCard[company="HYBE"] {
    border-left: 6px solid #f458b5;
}

QLabel#FrontendStatus {
    color: #ffffff;
    background: rgba(34, 16, 58, 150);
    border: 1px solid rgba(255, 255, 255, 95);
    border-radius: 8px;
    padding: 10px 12px;
    font-weight: 800;
}

QLabel#FrontendStage {
    background: rgba(9, 7, 20, 150);
    color: rgba(255, 255, 255, 226);
    border: 2px solid rgba(255, 255, 255, 82);
    border-radius: 8px;
    padding: 10px;
    font-size: 18px;
    font-weight: 900;
}

QPushButton#FrontendActionButton {
    color: #5d2fa6;
    background: rgba(255, 255, 255, 226);
    border: 2px solid rgba(255, 210, 255, 185);
    border-radius: 8px;
    padding: 12px 15px;
    font-weight: 900;
}

QPushButton#FrontendActionButton:disabled {
    color: rgba(60, 43, 79, 150);
    background: rgba(255, 255, 255, 128);
    border-color: rgba(255, 255, 255, 80);
}

QPushButton#FrontendStartButton {
    color: #ffffff;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff3fbf, stop:0.58 #9d56ff, stop:1 #57e9d7);
    border: 2px solid rgba(255, 255, 255, 195);
    border-radius: 8px;
    padding: 12px 20px;
    font-size: 18px;
    font-weight: 900;
}

QPushButton#FrontendStartButton:disabled {
    color: rgba(40, 35, 50, 150);
    background: rgba(255, 255, 255, 120);
    border-color: rgba(255, 255, 255, 70);
}

QCheckBox {
    color: #34312b;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 17px;
    height: 17px;
    border-radius: 4px;
    border: 1px solid #9f927d;
    background: #fffaf2;
}

QCheckBox::indicator:checked {
    background: #137c70;
    border-color: #0d665b;
}

QProgressBar {
    background: #eee4d5;
    border: 1px solid #d7c8b2;
    border-radius: 6px;
    color: #3f392f;
    height: 18px;
    text-align: center;
}

QProgressBar::chunk {
    background: #f0a431;
    border-radius: 5px;
}

QSlider::groove:horizontal {
    background: #ddd0bd;
    border-radius: 3px;
    height: 6px;
}

QSlider::handle:horizontal {
    background: #137c70;
    border: 1px solid #0d665b;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}
"""
