"""
download_semantickitti_seq00.py
--------------------------------
Downloads SemanticKITTI / KITTI Odometry velodyne scans for Sequence 00
and places the .bin files in ./data/sequences/00/velodyne/

HOW KITTI DOWNLOADS WORK
─────────────────────────
KITTI does NOT use username/password for file downloads.
Instead it emails you a time-limited direct download URL.

STEP 1 – Request the link (run this now):
    python download_semantickitti_seq00.py --request-link

    This submits your email to KITTI. Within a few minutes you will receive
    an email from KITTI with a direct download link.

STEP 2 – Download + extract (after you get the email):
    python download_semantickitti_seq00.py --url "<paste-link-from-email>"

ALTERNATIVE – If you already have the zip locally:
    python download_semantickitti_seq00.py --zip "C:/path/to/data_odometry_velodyne.zip"
"""

import argparse
import os
import sys
import zipfile
import shutil
import tempfile
import requests
from pathlib import Path
from tqdm import tqdm

EMAIL           = "25261999.rohan@gdgu.org"
REQUEST_URL     = "https://www.cvlibs.net/download.php?file=data_odometry_velodyne.zip"
SEQUENCE        = "00"
DEST_DIR        = Path("data") / "sequences" / SEQUENCE / "velodyne"
# Fixed cache dir – survives reboots; delete manually once extraction is done
CACHE_DIR       = Path("kitti_download_cache")


# ── Helpers ──────────────────────────────────────────────────────────────────

def request_download_link(email: str) -> None:
    """Submit the email form so KITTI emails a direct download link."""
    print(f"[*] Requesting download link from KITTI for {email} …")
    session = requests.Session()
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/124"
    )
    r = session.post(
        REQUEST_URL,
        data={
            "file": "data_odometry_velodyne.zip",
            "email": email,
            "submit": "Request Download Link",
        },
        timeout=30,
    )
    r.raise_for_status()
    if "sent" in r.text.lower() or "email" in r.text.lower() or r.status_code == 200:
        print("[+] Request submitted successfully!")
        print(f"    Check {email} for an email from no-reply@cvlibs.net")
        print("    (may take a few minutes — also check your spam folder)")
        print()
        print("Once you have the link, run:")
        print('    python download_semantickitti_seq00.py --url "<paste-link-here>"')
    else:
        print(f"[!] Unexpected response (status {r.status_code}). "
              f"Try visiting {REQUEST_URL} manually.")


def download_file(url: str, dest: Path) -> Path:
    """
    Stream-download with resume support (HTTP Range requests).
    If dest already exists with partial content, continues from where it left off.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0

    session = requests.Session()
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/124"
    )

    headers = {}
    if existing:
        headers["Range"] = f"bytes={existing}-"
        print(f"[*] Resuming from {existing / 1e9:.2f} GB …")
    else:
        print(f"[*] Starting fresh download …")
    print(f"    URL: {url}")
    print(f"    Saving to: {dest}")

    with session.get(url, stream=True, timeout=60, headers=headers) as r:
        if r.status_code == 416:  # Range not satisfiable – already complete
            print("[+] File already fully downloaded.")
            return dest
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        if "text/html" in ct:
            print("[!] The URL returned an HTML page, not a zip file.")
            print("    Check the URL and try again.")
            sys.exit(1)
        total_remaining = int(r.headers.get("content-length", 0))
        total = existing + total_remaining
        mode = "ab" if existing else "wb"
        with open(dest, mode) as f, tqdm(
            total=total, initial=existing,
            unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))

    print(f"[+] Download complete: {dest}")
    return dest


def extract_sequence(zip_path: Path, sequence: str, dest_dir: Path) -> int:
    """
    Extract only the velodyne .bin files for the given sequence.
    Returns number of files extracted.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    prefixes = [
        f"dataset/sequences/{sequence}/velodyne/",
        f"sequences/{sequence}/velodyne/",
        f"{sequence}/velodyne/",
    ]

    print(f"[*] Scanning archive for sequence {sequence} velodyne .bin files …")
    count = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = None
        for pfx in prefixes:
            candidates = [m for m in zf.namelist()
                          if m.startswith(pfx) and m.endswith(".bin")]
            if candidates:
                members = candidates
                print(f"[+] Found {len(members)} .bin files under '{pfx}'")
                break

        if not members:
            print("[!] Could not find sequence data in the archive.")
            print("    Entries sample:", zf.namelist()[:10])
            sys.exit(1)

        print(f"[*] Extracting {len(members)} files to {dest_dir} …")
        for member in tqdm(members, unit="file"):
            filename = Path(member).name
            target = dest_dir / filename
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1

    return count


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download SemanticKITTI Sequence 00 velodyne .bin files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--request-link", action="store_true",
                       help="Submit your email to KITTI to receive a download link")
    group.add_argument("--url", "-u",
                       help="Direct download URL received from KITTI email")
    group.add_argument("--zip", "-z",
                       help="Path to a locally available data_odometry_velodyne.zip")
    parser.add_argument("--email", "-e", default=EMAIL,
                        help=f"Email for --request-link (default: {EMAIL})")
    parser.add_argument("--dest", "-d", default=str(DEST_DIR),
                        help=f"Destination folder (default: {DEST_DIR})")
    args = parser.parse_args()

    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # ── Request link ─────────────────────────────────────────────────────────
    if args.request_link:
        request_download_link(args.email)
        return

    # ── Download from URL ────────────────────────────────────────────────────
    if args.url:
        print(f"[*] Destination: {dest_dir.resolve()}")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = CACHE_DIR / "data_odometry_velodyne.zip"
        print(f"[*] Cache dir (survives restarts): {CACHE_DIR.resolve()}")
        download_file(args.url, zip_path)
        count = extract_sequence(zip_path, SEQUENCE, dest_dir)
        print(f"\n[✓] Done – {count} .bin files extracted to:\n    {dest_dir.resolve()}")
        print(f"[*] You can delete the cache dir to free space:\n    {CACHE_DIR.resolve()}")
        return

    # ── Local zip ────────────────────────────────────────────────────────────
    if args.zip:
        zip_path = Path(args.zip)
        if not zip_path.is_file():
            print(f"[!] Zip not found: {zip_path}")
            sys.exit(1)
        print(f"[*] Destination: {dest_dir.resolve()}")
        count = extract_sequence(zip_path, SEQUENCE, dest_dir)
        print(f"\n[✓] Done – {count} .bin files extracted to:\n    {dest_dir.resolve()}")
        return

    # ── No args: print help ───────────────────────────────────────────────────
    parser.print_help()


if __name__ == "__main__":
    main()
