.PHONY: help clean build rust-build sdist wheel sha256 publish check sync release

# ============================================================
#  jinwu monorepo 一键构建与发布
#  发行包: packages/{jinwu,jinwu-ep,jinwu-swift,jinwu-fermi,jinwurs}
# ============================================================

PY_PACKAGES = packages/jinwu packages/jinwu-ep packages/jinwu-swift packages/jinwu-fermi

# 自动探测 Python 解释器（部分环境无裸 `python` 命令，只有 python3）
PYTHON := $(shell command -v python3 2>/dev/null || command -v python)

help:
	@echo "用法:"
	@echo "  make release   — 同步版本号 + git tag + push（唯一需要记的命令）"
	@echo "  make sync      — 从 packages/jinwu/pyproject.toml 同步版本号到 Cargo/recipe/仪器包"
	@echo "  make build     — 构建全部 Python wheel + sdist"
	@echo "  make clean     — 删除临时文件（保留 dist/ ）"
	@echo "  make check     — 运行测试"

# -----------------------------------------------------------
# 清理（只删临时文件，不动 dist/ 里的历史版本）
# -----------------------------------------------------------
clean:
	rm -rf build/ *.egg-info packages/*/build packages/*/*.egg-info
	rm -rf packages/jinwurs/target/wheels/
	@echo "✓ 临时文件已清理，dist/ 保留不动"

# -----------------------------------------------------------
# Rust 加速部分（可选 — 有 maturin 就编，没有就跳过）
# -----------------------------------------------------------
rust-build:
	@if command -v maturin >/dev/null 2>&1; then \
		echo "→ 编译 Rust 扩展 (jinwurs)..."; \
		cd packages/jinwurs && maturin build --release --out ../../dist/ && cd ../.. ; \
		echo "✓ jinwurs wheel 已生成到 dist/"; \
	else \
		echo "ℹ maturin 未安装，跳过 Rust 扩展编译"; \
	fi

# -----------------------------------------------------------
# Python 构建（遍历所有发行包）
# -----------------------------------------------------------
sdist:
	@for pkg in $(PY_PACKAGES); do \
		echo "→ sdist: $$pkg"; \
		$(PYTHON) -m build --sdist "$$pkg" --outdir dist/ || exit 1; \
	done

wheel:
	@for pkg in $(PY_PACKAGES); do \
		echo "→ wheel: $$pkg"; \
		$(PYTHON) -m build --wheel "$$pkg" --outdir dist/ || exit 1; \
	done

build: sdist wheel
	@echo "✓ 构建完成:"
	@ls -lh dist/

# -----------------------------------------------------------
# 更新 meta.yaml 里的 sha256（只处理核心包）
# -----------------------------------------------------------
sha256:
	@LATEST=$$(ls -t dist/jinwu-*.tar.gz 2>/dev/null | grep -v jinwu- | head -1); \
	LATEST=$$(ls -t dist/jinwu-[0-9]*.tar.gz 2>/dev/null | head -1); \
	if [ -z "$$LATEST" ]; then \
		echo "❌ 没有找到核心包 sdist，请先运行 make build"; \
		exit 1; \
	fi; \
	HASH=$$(openssl sha256 "$$LATEST" | awk '{print $$2}'); \
	VERSION=$$($(PYTHON) -c "import tomllib; print(tomllib.load(open('packages/jinwu/pyproject.toml','rb'))['project']['version'])"); \
	sed -i "s/^  version:.*/  version: \"$$VERSION\"/" recipe/meta.yaml; \
	sed -i "s/^  sha256:.*/  sha256: $$HASH/" recipe/meta.yaml; \
	echo "✓ recipe/meta.yaml 更新完毕:"; \
	echo "  version = $$VERSION"; \
	echo "  sha256  = $$HASH"

# -----------------------------------------------------------
# 上传到 PyPI（全部发行包）
# -----------------------------------------------------------
publish: build sha256
	@echo "→ 上传到 PyPI..."
	$(PYTHON) -m twine upload dist/*.tar.gz dist/*-py3-none-any.whl
	@echo "✓ 发布完成！"

# -----------------------------------------------------------
# 测试
# -----------------------------------------------------------
check:
	$(PYTHON) -m pytest test/ -x -q

# -----------------------------------------------------------
# 从 packages/jinwu/pyproject.toml 读取版本号，同步到所有文件
# -----------------------------------------------------------
_VERSION = $(shell $(PYTHON) -c "import tomllib; print(tomllib.load(open('packages/jinwu/pyproject.toml','rb'))['project']['version'])")

sync:
	@echo "→ 从 packages/jinwu/pyproject.toml 读取版本号: $(_VERSION)"
	@# recipe/meta.yaml
	@sed -i 's/{% set version = ".*" %}/{% set version = "$(_VERSION)" %}/' recipe/meta.yaml
	@sed -i 's/version: ".*"/version: "$(_VERSION)"/' recipe/meta.yaml
	@# Cargo.toml / pyproject.toml (jinwurs)
	@sed -i 's/^version = ".*"/version = "$(_VERSION)"/' packages/jinwurs/Cargo.toml
	@sed -i 's/^version = ".*"/version = "$(_VERSION)"/' packages/jinwurs/pyproject.toml
	@# 仪器包版本（锁步发布）
	@for pkg in packages/jinwu-ep packages/jinwu-swift packages/jinwu-fermi; do \
		sed -i 's/^version = ".*"/version = "$(_VERSION)"/' "$$pkg/pyproject.toml"; \
	done
	@echo "✓ 全部同步到 $(_VERSION)"

# ── 一键发布 ──────────────────────────────────────────────
release: sync
	@echo ""
	@echo "→ 提交变更..."
	git add packages/jinwu/pyproject.toml packages/jinwu-ep/pyproject.toml \
		packages/jinwu-swift/pyproject.toml packages/jinwu-fermi/pyproject.toml \
		packages/jinwurs/Cargo.toml packages/jinwurs/pyproject.toml packages/jinwurs/README.md \
		recipe/meta.yaml
	git diff --cached --stat
	@echo ""
	git commit -m "release: jinwu v$(_VERSION)"
	@echo ""
	@echo "→ git tag v$(_VERSION) ..."
	git tag v$(_VERSION)
	git push origin beta
	git push origin v$(_VERSION)
	@echo ""
	@echo "✓ 已推送 v$(_VERSION) → GitHub Actions 自动构建发布"
