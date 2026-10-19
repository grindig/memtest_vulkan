//! Console width at startup and text fitting, so report lines don't wrap mid-word in the window.
use std::sync::OnceLock;

/// Width used when stdout isn't a console (redirected to a file or pipe) and COLUMNS isn't set.
const FALLBACK_COLUMNS: usize = 120;
/// Narrower windows still get the time, status and speed columns.
const MIN_COLUMNS: usize = 40;

/// Width and where it came from, measured once when first asked for.
fn startup_width() -> (usize, &'static str) {
    static WIDTH: OnceLock<(usize, &'static str)> = OnceLock::new();
    *WIDTH.get_or_init(|| {
        let (columns, source) = if let Some(columns) = window_columns() {
            (columns, "console window")
        } else if let Some(columns) = std::env::var("COLUMNS").ok().and_then(|c| c.parse().ok()) {
            (columns, "COLUMNS variable")
        } else {
            (FALLBACK_COLUMNS, "default, output is not a console")
        };
        (columns.max(MIN_COLUMNS), source)
    })
}

/// Visible width of the console window when the process started. Later resizes are ignored.
pub fn columns() -> usize {
    startup_width().0
}

/// e.g. "120 columns (console window)", for verbose output
pub fn describe() -> String {
    let (columns, source) = startup_width();
    format!("{columns} columns ({source})")
}

/// Usable line length: printing into the last column makes some consoles add an empty line.
pub fn line_width() -> usize {
    columns() - 1
}

#[cfg(windows)]
fn window_columns() -> Option<usize> {
    use windows_sys::Win32::System::Console::{
        CONSOLE_SCREEN_BUFFER_INFO, GetConsoleScreenBufferInfo, GetStdHandle, STD_OUTPUT_HANDLE,
    };
    let mut info: CONSOLE_SCREEN_BUFFER_INFO = unsafe { std::mem::zeroed() };
    // the visible window, not the buffer: legacy consoles can have a wider buffer than window
    let ok = unsafe { GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &mut info) };
    let width = i32::from(info.srWindow.Right) - i32::from(info.srWindow.Left) + 1;
    (ok != 0 && width > 0).then_some(width as usize)
}

#[cfg(not(windows))]
fn window_columns() -> Option<usize> {
    let size = mortal::Terminal::new().ok()?.size().ok()?;
    (size.columns > 0).then_some(size.columns)
}

/// Word-wraps `text` to `width` characters per line, indenting continuation lines by two spaces.
/// Existing line breaks are kept and words longer than a line are not split.
pub fn wrap(text: &str, width: usize) -> String {
    let mut lines = Vec::new();
    for paragraph in text.split('\n') {
        let mut line = String::new();
        for word in paragraph.split_whitespace() {
            let has_text = !line.trim().is_empty();
            if has_text && line.chars().count() + 1 + word.chars().count() > width {
                lines.push(std::mem::replace(&mut line, format!("  {word}")));
            } else {
                if has_text {
                    line.push(' ');
                }
                line.push_str(word);
            }
        }
        lines.push(line);
    }
    lines.join("\n")
}

/// A table column: header, width in characters and alignment.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Column {
    pub header: &'static str,
    pub width: usize,
    pub left_aligned: bool,
}

const COLUMN_GAP: usize = 2;

/// How many of the leading `columns` fit in `width`, with COLUMN_GAP spaces between columns.
pub fn fitting_columns(columns: &[Column], width: usize) -> usize {
    let mut used = 0;
    columns
        .iter()
        .take_while(|column| {
            let next = used + if used == 0 { 0 } else { COLUMN_GAP } + column.width;
            let fits = next <= width;
            if fits {
                used = next;
            }
            fits
        })
        .count()
}

/// One table line; values longer than their column are cut to the column width.
pub fn row(columns: &[Column], values: &[String]) -> String {
    let cells: Vec<String> = columns
        .iter()
        .zip(values)
        .map(|(column, value)| {
            let value: String = value.chars().take(column.width).collect();
            let width = column.width;
            if column.left_aligned {
                format!("{value:<width$}")
            } else {
                format!("{value:>width$}")
            }
        })
        .collect();
    cells.join(&" ".repeat(COLUMN_GAP)).trim_end().to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_breaks_at_words_and_indents_continuations() {
        let text = "Extended mode: 6 data-pattern families, March C- on every pass";
        assert_eq!(
            wrap(text, 30),
            "Extended mode: 6 data-pattern\n  families, March C- on every\n  pass"
        );
        for line in wrap(text, 30).lines() {
            assert!(line.chars().count() <= 30, "{line:?}");
        }
        assert_eq!(wrap("short\nlines stay", 30), "short\nlines stay");
        assert_eq!(
            wrap("a verylongwordthatcannotsplit b", 10),
            "a\n  verylongwordthatcannotsplit\n  b"
        );
    }

    #[test]
    fn columns_are_dropped_from_the_end_when_narrow() {
        let columns = [
            Column {
                header: "time",
                width: 8,
                left_aligned: true,
            },
            Column {
                header: "read",
                width: 6,
                left_aligned: false,
            },
            Column {
                header: "power",
                width: 5,
                left_aligned: false,
            },
        ];
        assert_eq!(fitting_columns(&columns, 23), 3); // 8 + 2 + 6 + 2 + 5
        assert_eq!(fitting_columns(&columns, 22), 2);
        assert_eq!(fitting_columns(&columns, 7), 0);
        let values = ["00:00:36".to_owned(), "697.7".to_owned(), "262W".to_owned()];
        assert_eq!(row(&columns, &values), "00:00:36   697.7   262W");
        assert_eq!(
            row(&columns[..1], &["too long for it".to_owned()]),
            "too long"
        );
    }
}
