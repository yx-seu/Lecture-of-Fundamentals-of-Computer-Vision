#!/usr/bin/env python3
"""
build_app.py — Build LeNet-5 application on existing platform
Usage: vitis -s build_app.py
"""
import os, sys

WS   = "/home/aika/cv/sw/vitis_ws"
SRC  = "/home/aika/cv/sw/src/main_demo.c"
APP  = "lenet5_demo"

def main():
    import vitis
    client = vitis.create_client()
    client.set_workspace(WS)

    # Platform already built — skip creation
    print(f"Workspace: {WS}")

    # Check existing platform
    platforms = client.list_platforms()
    print(f"Existing platforms: {platforms}")

    if not platforms:
        print("ERROR: No platform found. Run build_vitis_v2.py first.")
        return 1

    plat_path = platforms[0][0]  # (path, type, arch)
    print(f"Using platform: {plat_path}")

    # Create application
    print(f"\nCreating application '{APP}'...")
    app = client.create_app_component(
        name=APP,
        platform=plat_path,
        domain="standalone_domain"
    )

    # Import source
    print(f"Importing source: {SRC}")
    try:
        app.import_sources(SRC, "main.c")
    except AttributeError:
        # Copy manually
        dest = os.path.join(WS, APP, "src")
        os.makedirs(dest, exist_ok=True)
        with open(SRC) as f:
            src_text = f.read()
        with open(os.path.join(dest, "main.c"), "w") as f:
            f.write(src_text)
        print(f"  Copied to {dest}/main.c")

    # Build
    print("Building...")
    app.build()

    # Find ELF
    for root, dirs, files in os.walk(os.path.join(WS, APP)):
        for f in files:
            if f.endswith('.elf'):
                elf = os.path.join(root, f)
                print(f"\n BUILD SUCCESS!")
                print(f" ELF: {elf}")
                import shutil
                shutil.copy(elf, "/home/aika/cv/sw/lenet5_demo.elf")
                print(f" Copied to: /home/aika/cv/sw/lenet5_demo.elf")
                client.close()
                return 0

    print("Build completed. Check workspace for ELF.")
    client.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())
