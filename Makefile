.PHONY: help clean build rust-build sdist wheel sha256 publish check sync release

# ============================================================
#  jinwu 一键构建与发布
# ============================================================

help:
	@echo "用法:"
	@echo "  make release   — 同步版本号 + git tag + push（唯一需要记的命令）"
	@echo "  make sync      — 从 pyproject.toml 同步版本号到 Cargo/recipe"
	@echo "  make build     — 构建 Python wheel + sdist"
	@echo "  make clean     — 删除临时文件（保留 dist/ ）"
	@echo "  make check     — 运行测试"

# -----------------------------------------------------------
# 清理（只删临时文件，不动 dist/ 里的历史版本）
# -----------------------------------------------------------
clean:
	rm -rf build/ *.egg-info src/*.egg-info
	rm -rf src/jinwurs/target/wheels/
	@echo "✓ 临时文件已清理，dist/ 保留不动"

# -----------------------------------------------------------
# Rust 加速部分（可选 — 有 maturin 就编，没有就跳过）
# -----------------------------------------------------------
rust-build:
	@if command -v maturin >/dev/null 2>&1; then \
		echo "→ 编译 Rust 扩展 (jinwurs)..."; \
		cd src/jinwurs && maturin build --release --out ../../dist/ && cd ../.. ; \
		echo "✓ jinwurs wheel 已生成到 dist/"; \
	else \
		echo "ℹ maturin 未安装，跳过 Rust 扩展编译"; \
	fi

# -----------------------------------------------------------
# Python 构建
# -----------------------------------------------------------
sdist:
	python -m build --sdist

wheel:
	python -m build --wheel

build: sdist wheel
	@echo "✓ 构建完成:"
	@ls -lh dist/

# -----------------------------------------------------------
# 更新 meta.yaml 里的 sha256
# -----------------------------------------------------------
sha256:
	@LATEST=$$(ls -t dist/jinwu-*.tar.gz 2>/dev/null | head -1); \
	if [ -z "$$LATEST" ]; then \
		echo "❌ 没有找到 sdist，请先运行 make build"; \
		exit 1; \
	fi; \
	HASH=$$(openssl sha256 "$$LATEST" | awk '{print $$2}'); \
	VERSION=$$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])"); \
	sed -i "s/^  version:.*/  version: \"$$VERSION\"/" recipe/meta.yaml; \
	sed -i "s/^  sha256:.*/  sha256: $$HASH/" recipe/meta.yaml; \
	echo "✓ recipe/meta.yaml 更新完毕:"; \
	echo "  version = $$VERSION"; \
	echo "  sha256  = $$HASH"

# -----------------------------------------------------------
# 上传到 PyPI
# -----------------------------------------------------------
publish: build sha256
	@echo "→ 上传到 PyPI..."
	python -m twine upload dist/jinwu-*.tar.gz dist/jinwu-*-py3-none-any.whl
	@echo "✓ 发布完成！"

# -----------------------------------------------------------
# 测试
# -----------------------------------------------------------
check:
	python -m pytest test/ -x -q

# -----------------------------------------------------------
# 从 pyproject.toml 读取版本号，同步到所有文件
# -----------------------------------------------------------
_VERSION = $(shell python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")

sync:
	@echo "→ 从 pyproject.toml 读取版本号: $(_VERSION)"
	@# recipe/meta.yaml
	@sed -i 's/{% set version = ".*" %}/{% set version = "$(_VERSION)" %}/' recipe/meta.yaml
	@sed -i 's/version: ".*"/version: "$(_VERSION)"/' recipe/meta.yaml
	@# Cargo.toml (jinwurs)
	@sed -i 's/^version = ".*"/version = "$(_VERSION)"/' src/jinwurs/Cargo.toml
	@# pyproject.toml (jinwurs)
	@sed -i 's/^version = ".*"/version = "$(_VERSION)"/' src/jinwurs/pyproject.toml
	@echo "✓ 全部同步到 $(_VERSION)"

# ── 一键发布 ──────────────────────────────────────────────
release: sync
	@echo ""
	@echo "→ 提交变更..."
	git add pyproject.toml recipe/meta.yaml src/jinwurs/Cargo.toml src/jinwurs/pyproject.toml
	git diff --cached --stat
	@echo ""
	git commit -m "release: jinwu v$(_VERSION)"
	@echo ""
	@echo "→ git tag v$(_VERSION) ..."
	git tag v$(_VERSION)
	git push origin master
	git push origin v$(_VERSION)
	@echo ""
	@echo "✓ 已推送 v$(_VERSION) → GitHub Actions 自动构建发布"
