"""AppleSupport Intent Taxonomy Definition.

Consolidated 7 technical intents + 1 catch-all (other_unclear) derived
empirically from clustering 85,000+ historical AppleSupport tweets.
"""

from __future__ import annotations

import json
import os

TAXONOMY = {
    "battery_power_issue": {
        "name": "battery_power_issue",
        "display_name": "Battery & Power Degradation",
        "description": "Rapid battery drain, iPhone/Mac dying unexpectedly, charging issues, battery health drop after iOS update.",
        "exemplars": [
            "My iPhone 7 battery is draining so fast since iOS 11.1, barely lasts 3 hours.",
            "Watching the battery percentage drop like a second counter on my iPhone 8.",
            "Phone shuts down at 20% battery remaining and won't turn back on without charger."
        ],
        "default_auto_handle": True
    },
    "os_software_glitch": {
        "name": "os_software_glitch",
        "display_name": "OS Updates & System Bugs",
        "description": "iOS update errors, keyboard typing bugs (e.g. 'i' autocorrect to A?), app crashes, boot loops, system freezing.",
        "exemplars": [
            "Your newest update glitch of replacing the 'I' with 'A?' is making me mad.",
            "Ever since updating to iOS 11.0.2 my phone has been freezing and lagging constantly.",
            "My phone is stuck on the Apple logo boot loop after attempting the latest update."
        ],
        "default_auto_handle": True
    },
    "apps_services_media": {
        "name": "apps_services_media",
        "display_name": "Apps, Media & Apple Services",
        "description": "Apple Music library wiped/syncing, App Store download errors, iMessage/FaceTime activation failures, Safari issues.",
        "exemplars": [
            "Why did all of my Apple Music library erase once I updated my 6S Plus?",
            "What is going on with iMessage and FaceTime? They are not working on my iPad.",
            "App Store gives me an error code 600 whenever I try to download or update apps."
        ],
        "default_auto_handle": True
    },
    "connectivity_network": {
        "name": "connectivity_network",
        "display_name": "Connectivity, Wi-Fi & Bluetooth",
        "description": "Wi-Fi dropping or refusing to connect, Bluetooth pairing failures (AirPods, car), cellular data signal loss.",
        "exemplars": [
            "My iPhone won't connect to any Wi-Fi network even though all my other devices work.",
            "Bluetooth keeps disconnecting from my AirPods every 2 minutes.",
            "No service / Searching error on my iPhone after traveling abroad."
        ],
        "default_auto_handle": True
    },
    "account_icloud_security": {
        "name": "account_icloud_security",
        "display_name": "Apple ID, iCloud & Account Security",
        "description": "Apple ID locked for security reasons, 2FA code not received, iCloud backup failing or out of storage, suspicious emails.",
        "exemplars": [
            "My Apple ID has been locked for security reasons and I can't access my iCloud.",
            "Not receiving two-factor authentication verification code on my trusted number.",
            "Just got email asking for my order number and password, is this phishing or legit?"
        ],
        "default_auto_handle": False  # Security and account lockouts often need sensitive verification
    },
    "hardware_screen_audio": {
        "name": "hardware_screen_audio",
        "display_name": "Hardware, Audio & Physical Damage",
        "description": "Broken/unresponsive touchscreen, speaker crackling, microphone not picking up audio, water damage, swollen battery.",
        "exemplars": [
            "My iPhone speaker just stopped working and crackles during phone calls.",
            "Half of my screen is black with lines and touch is completely unresponsive.",
            "My phone fell in water and now the home button won't click."
        ],
        "default_auto_handle": False  # Physical repairs require Genius Bar appointment / mail-in
    },
    "orders_repairs_store": {
        "name": "orders_repairs_store",
        "display_name": "Orders, Store Appointments & Warranty",
        "description": "Apple Store Genius Bar reservations, repair status check, order shipment delays (iPhone X delivery), trade-in values.",
        "exemplars": [
            "How do I book a Genius Bar appointment at the Regent Street Apple Store?",
            "Ordered an iPhone X on launch day and shipping status hasn't updated in 2 weeks.",
            "What is the trade-in credit for an iPhone 6 in working condition?"
        ],
        "default_auto_handle": True
    },
    "theft_lost_legal": {
        "name": "theft_lost_legal",
        "display_name": "Theft, Stolen Devices & Urgent Escalation",
        "description": "Stolen Mac/iPhone, tracking stolen device with police, harassment, explicit legal / lawsuit threats, abusive language.",
        "exemplars": [
            "My MacBook was stolen yesterday, police report filed, can you help track the serial number?",
            "I will be filing a lawsuit against Apple if my issue is not resolved by a human manager today.",
            "Someone stole my iPhone and is trying to bypass iCloud activation lock."
        ],
        "default_auto_handle": False  # Mandatory immediate escalation
    },
    "other_unclear": {
        "name": "other_unclear",
        "display_name": "General Inquiries / Unclear Chatter",
        "description": "General rants without specific technical details, emojis only, non-English tweets, or non-actionable banter.",
        "exemplars": [
            "Apple is the worst company ever smh 🙄",
            "Menuda santa mierda la última actualización de IOS",
            "Can someone help me please thanks"
        ],
        "default_auto_handle": True
    }
}

INTENT_NAMES = list(TAXONOMY.keys())

def save_taxonomy_json(out_path: str = "data/taxonomy.json"):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(TAXONOMY, f, indent=2)
    print(f"Taxonomy saved to {out_path} ({len(TAXONOMY)} intents)")

if __name__ == "__main__":
    save_taxonomy_json()
