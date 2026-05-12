# Sync local repo (e.g. F:) to Google Drive

Two separate `!python` runs on Colab **do not share RAM**, so code changes must reach Drive (or your Colab copy) somehow. These options keep **Google Drive** in sync with a folder on your **F:** drive without re-uploading the whole tree every time.

## 1. Google Drive for desktop (simplest)

1. Install [Google Drive for desktop](https://support.google.com/drive/answer/2374987) on the PC that has **F:**.
2. Sign in with the same Google account you use in Colab.
3. In Drive settings, choose **Stream** or **Mirror** for `My Drive`.
4. Either:
   - **Put the repo inside the synced Drive folder** (e.g. move or clone `imagination-v1.1.0` under `G:\My Drive\` or the mounted path Drive shows), and use **F:** only as a duplicate via symlink/copy, or  
   - **Map a folder**: create `My Drive/imagination-v1.1.0` and use Explorer to work from that path (it may appear as a drive letter like `G:`).

5. After you save files in Cursor on that path, Drive syncs in the background. On Colab, your next session sees updated **code** immediately. **Large `.safetensors`** files also sync but are slow the first time; avoid editing them frequently.

**Caveat:** Sync conflicts if you edit the same file on two machines while offline. Close Colab or stop editing on one side while syncing.

## 2. Stream files from F: into Drive (advanced)

If the repo must stay on **F:** only:

- Use **Drive for desktop** “mirror” of a folder, or  
- **rclone** one-way sync:  
  `rclone sync "F:/imagination-v1.1.0" remote:imagination-v1.1.0`  
  (configure `remote` as Google Drive once). Run manually or on a **Task Scheduler** job every N minutes.

Exclude huge folders from sync if you only change code:

```text
--exclude "*.safetensors"
```

(Only if weights on Colab come from another copy or download.)

## 3. Git instead of file sync

- Push from **F:** to GitHub; on Colab `git clone` / `git pull` into `/content`.  
- **Weights** stay on Drive or LFS; repo stays small. Best when you change code often.

## Summary

| Goal | Approach |
|------|----------|
| Easiest “save on F: and see in Colab” | Work inside **Drive for desktop** folder, or sync F: → Drive with rclone |
| Only sync small files | rclone with excludes, or Git for code + Drive for weights |

Imagination Colab paths expect something like:  
`My Drive/imagination-v1.1.0` → symlink to `/content/imagination-v1.1.0` (see `COLAB_SETUP.md`).
