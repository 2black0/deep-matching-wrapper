#!/bin/bash
# Build all matcher-cpp modules
# Usage: bash scripts/build_all_matchers_cpp.sh

set -e

MATCHERS="xfeat liftfeat clidd"

echo "======================================================="
echo "🔨 Building all matcher-cpp modules"
echo "======================================================="

for MATCHER in $MATCHERS; do
    echo ""
    bash scripts/build_matcher_cpp.sh "$MATCHER"
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to build $MATCHER"
        exit 1
    fi
done

echo ""
echo "======================================================="
echo "✅ All matchers built successfully!"
echo "======================================================="
echo ""
echo "Built matchers:"
for MATCHER in $MATCHERS; do
    if [ -f "matcher-cpp/$MATCHER/build/demo_$MATCHER" ]; then
        echo "  ✅ $MATCHER: matcher-cpp/$MATCHER/build/demo_$MATCHER"
    else
        echo "  ❌ $MATCHER: Binary not found!"
    fi
done
