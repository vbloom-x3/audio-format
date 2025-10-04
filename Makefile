# Compiler
CC = gcc

# Use pkg-config to get cflags and libs for libvlc
CFLAGS = -O3
LDFLAGS = -lsndfile -lm

# Source and output
SRC = src/main.c
OUT = codec

# Default rule
all: $(OUT)

# Linking
$(OUT): $(SRC)
	$(CC) $(CFLAGS) -o $(OUT) $(SRC) $(LDFLAGS)
	@echo "Done!"

# Clean rule
clean:
	rm -f $(OUT)

install: $(OUT)
	mkdir -p ~/.local/bin
	cp $(OUT) ~/.local/bin/libblos
	@echo -e "\e[31m[WARN]\e[0m : Please make sure that ~/.local/bin/ is in your PATH"
	@echo "Do you want to add ~/.local/bin/ to your PATH? (y/n)"; \
	read -p ":" choice; \
	if [ "$$choice" = "y" ] || [ "$$choice" = "Y" ]; then \
		echo "You have chosen 'yes'."; \
		echo "Adding ~/.local/bin to PATH permanently..."; \
		if [ -f "$$HOME/.bashrc" ]; then \
			echo 'export PATH="$$HOME/.local/bin:$$PATH"' >> ~/.bashrc; \
		fi; \
		if [ -f "$$HOME/.zshrc" ]; then \
			echo 'export PATH="$$HOME/.local/bin:$$PATH"' >> ~/.zshrc; \
		fi; \
		if [ -d "$$HOME/.config/fish" ]; then \
			echo 'set -U fish_user_paths $$HOME/.local/bin $$fish_user_paths' >> ~/.config/fish/config.fish; \
		fi; \
		if [ -f "$$HOME/.profile" ]; then \
			echo 'export PATH="$$HOME/.local/bin:$$PATH"' >> ~/.profile; \
		fi; \
		echo "Done! Please restart your terminal or run 'source ~/.bashrc', 'source ~/.zshrc', or 'source ~/.profile' to apply changes."; \
	else \
		echo "You have chosen 'no'."; \
	fi