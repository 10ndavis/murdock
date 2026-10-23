#!/usr/bin/env bash
# Runs EchoBrain's real startup code on a desktop JVM against a stand-in LiteRT
# (Fakes.kt), across two "launches" -- the second with saved learning present, which is
# where a Kotlin initialisation-order bug crashed the app on the phone.
#   KOTLINC=/path/to/kotlinc tool/jvm_startup_check/run.sh
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
src="$here/../../android/app/src/main/kotlin/com/murdock/murdock"
out=$(mktemp -d)
"${KOTLINC:-kotlinc}" -jvm-target 17 -nowarn -d "$out" \
  "$here/Fakes.kt" "$here/android/util/Log.kt" "$src/EchoBrain.kt" "$src/ShapeFit.kt" "$src/SonarConfig.kt" "$here/StartupTest.kt"
java -cp "$out:$(dirname "$(command -v "${KOTLINC:-kotlinc}")")/../lib/kotlin-stdlib.jar" StartupTestKt
