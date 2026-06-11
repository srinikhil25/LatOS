// Latos desktop shell — Tauri 2.
//
// The shell stays deliberately thin: native window, file dialogs, and
// (later) spawning the Python sidecar. All application logic lives in
// the React frontend + the latos-core sidecar API on 127.0.0.1.

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
