// main.c
// Compile: gcc main.c -o codec -lsndfile -lm -O3
// Usage:
//   Encode: ./codec input.wav output.blos
//   Decode: ./codec -d input.blos output.wav
// Improvements:
//  - float32 LPC coeffs (less quantization error)
//  - per-sample transient bypass (writes raw sample when residual too large)
//  - Rice coding for residuals
//  - frame-based, bit-packed, aligned

// Annotated hex-dump:

// -----------------------------
// Example BLOS file annotated hex
// -----------------------------

// 00000000: 42 4C 4F 53   D4 AC 00 00   30 79 0A 01  02 00 02 00
//           ^^^^           ^^^^          ^^ ^^ ^^ ^^
//           "BLOS" magic   unknown      sample rate (little-endian)

// 00000010: 00 3D 85 00   00 00 02 00   00 04 00 00  00 00 00 00
//           ^^^^           ^^^^
//           total frames   ????

// 00000020: FF FF 02 00   01 00 FF FF   01 00 00 00  3B 77 B6 E3
//           ^^^^           ^^^^
//           channels?      frame size?

// 00000030: DD 2C E7 BF   68 20 0E 55   95 AC C3 BF  93 3A 45 B6
//           ^^^^^^^^^^^^^^^^
//           LPC coefficients (double float?)

// 00000040: 99 63 9E 3F   7E 35 15 E1   56 E7 81 3F  D4 53 0E AF
//           ^^^^^^^^^^^^^^^^
//           residuals / seeds

// 00000050: B8 29 F5 BF   0F 6A 4A FF   72 9E F0 BF  A5 9C A6 0E
//           ^^^^^^^^^^^^^^^^
//           more residuals / seeds

// 00000060: 4F 1C DA BF   8C 74 EB CA   37 97 A3 BF  18 40 00 00
//           ^^^^^^^^^^^^^^^^
//           more residuals

// 00000070: 00 0C 20 00   00 00 01 0C   00 02 18 00  01 00 0C 00
//           ^^^^^^^^^^^^^^^^
//           header / alignment / padding

// 00000080: 00 00 86 19   86 4C 00 21   90 C0 C0 21  84 C0 20 00
//           ^^^^^^^^^^^^^^^^
//           more doubles / residuals

// The next section is by an A.G.I. chatbot, will be refined later or changed as per needs.

// -----------------------------
// Global header (first 32 bytes in file)
// -----------------------------
// "BLOS" magic        -> 4 bytes: identifies the file format
// Sample rate          -> 4 bytes (uint32_t), e.g., 44100 Hz
// Total frames         -> 4 bytes (uint32_t), total samples or frames × channels
// Channels             -> 1 byte (uint8_t), usually 2 for stereo
// Frame size           -> 4 bytes (uint32_t), e.g., 512 samples per frame
// Number of frames     -> 4 bytes (uint32_t), (frames_total + frame_size-1)/frame_size
//
// Little-endian format is used for all multi-byte numbers.

// -----------------------------
// Frame header
// -----------------------------
// Frame size (fsize)   -> 4 bytes (uint32_t), samples in this frame
// LPC order            -> 1 byte (uint8_t), prediction order
// Rice kA              -> 1 byte (uint8_t), Rice parameter for M channel residuals
// Rice kB              -> 1 byte (uint8_t), Rice parameter for S channel residuals
// (Optional padding)   -> to align doubles, may contain 0x00 or 0xFFFF

// -----------------------------
// Seeds (initial samples for reconstruction)
// -----------------------------
// For each channel (M and S), first `order` samples are stored as int16_t (2 bytes each)
// These initialize the reconstructed frame samples before LPC prediction begins

// -----------------------------
// LPC coefficients
// -----------------------------
// Each channel has `order` coefficients stored as doubles (8 bytes each)
// Stored after seeds. Used for predicting each sample from previous reconstructed samples
// Doubles reduce quantization error compared to float32

// -----------------------------
// Residuals (prediction errors)
// -----------------------------
// Stored after LPC coefficients, per sample from n=order..fsize-1
// Each residual has a 1-bit escape flag:
//   0 -> Rice-coded residual (using kA/kB)
//   1 -> Raw 16-bit sample (if residual magnitude > ESCAPE_THRESHOLD)
// Residuals + escape flags are written using the BitWriter, aligned to byte boundary

// -----------------------------
// M/S -> L/R reconstruction
// -----------------------------
// During decoding, M/S residuals + coefficients + seeds are used to reconstruct 
// Mframe and Sframe sample-by-sample. Then converted to L/R samples:
// L = M + S/2, R = M - S/2, clamped to int16_t

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <sndfile.h>
#include <limits.h>

// these can be tinkered with!
#define MAGIC "BLOS"

// can reduce or decrease for better / worse compression
#define FRAME_SIZE 512
#define MIN_ORDER 2
#define MAX_ORDER 16
#define DEFAULT_SCALE 16384
#define PROGRESS_STEP 4

// threshold for bypassing (tuneable). If abs(residual) > ESCAPE_THRESHOLD, we store raw sample.
#define ESCAPE_THRESHOLD 20000

// ----------------------------- 
// BitWriter / BitReader
// -----------------------------

typedef struct { uint8_t buf; int used; FILE *f; } BitWriter;
typedef struct { uint8_t buf; int left; FILE *f; } BitReader;

static inline void bw_init(BitWriter *bw, FILE *f) { bw->buf = 0; bw->used = 0; bw->f = f; }

static inline void bw_write_bit(BitWriter *bw, int bit) {
    bw->buf = (bw->buf << 1) | (bit & 1);
    bw->used++;
    if (bw->used == 8) { fputc(bw->buf, bw->f); bw->used = 0; bw->buf = 0; }
}

static inline void bw_write_bits(BitWriter *bw, uint32_t value, int nbits) {
    for (int i = nbits - 1; i >= 0; --i) bw_write_bit(bw, (value >> i) & 1);
}

static inline void bw_flush(BitWriter *bw) {
    if (bw->used > 0) { bw->buf <<= (8 - bw->used); fputc(bw->buf, bw->f); bw->used = 0; bw->buf = 0; }
}

static inline void br_init(BitReader *br, FILE *f) { br->buf = 0; br->left = 0; br->f = f; }

static inline int br_read_bit(BitReader *br) {
    if (br->left == 0) {
        int c = fgetc(br->f);
        if (c == EOF) return -1;
        br->buf = (uint8_t)c;
        br->left = 8;
    }
    int bit = (br->buf >> (br->left - 1)) & 1;
    br->left--;
    return bit;
}

static inline int br_read_bits(BitReader *br, int nbits, uint32_t *out) {
    uint32_t v = 0;
    for (int i = 0; i < nbits; ++i) {
        int b = br_read_bit(br);
        if (b < 0) return -1;
        v = (v << 1) | (b & 1);
    }
    *out = v;
    return 0;
}

static inline void br_byte_align(BitReader *br) { br->left = 0; }

// ----------------------------- 
// Levinson-Durbin Linear Predictive Coding Algorithm 
// -----------------------------

static void levinson_durbin(const double *r, int order, double *a_out) {
    double *a = (double*)calloc(order+1, sizeof(double));
    double *e = (double*)calloc(order+1, sizeof(double));
    a[0] = 1.0;
    e[0] = (r[0] > 1e-12) ? r[0] : 1e-12;
    for (int i = 1; i <= order; ++i) {
        double acc = 0.0;
        for (int j = 1; j < i; ++j) acc += a[j] * r[i - j];
        double k = (r[i] - acc) / e[i - 1];
        a[i] = k;
        for (int j = 1; j < i; ++j) a[j] = a[j] - k * a[i - j];
        e[i] = e[i - 1] * (1.0 - k * k);
        if (e[i] <= 1e-12) e[i] = 1e-12;
    }
    for (int i = 0; i < order; ++i) a_out[i] = a[i + 1];
    free(a); free(e);
}

// ----------------------------- 
// autocorrect 
// -----------------------------

static void compute_autocorr_frame(const double *x, int N, int order, double *r_out) {
    for (int k = 0; k <= order; ++k) {
        double s = 0.0;
        for (int n = k; n < N; ++n) s += x[n] * x[n - k];
        r_out[k] = s;
    }
}

// -----------------------------
// compute LPC & residuals
// -----------------------------

static void compute_lpc_and_residuals_frame(const int16_t *signal, int N, int order,
                                            double *coeffs_out, int32_t *residuals_out, int16_t *seeds_out,
                                            double *scratch_x, double *scratch_r) {
    for (int i = 0; i < N; ++i) scratch_x[i] = (double)signal[i];
    compute_autocorr_frame(scratch_x, N, order, scratch_r);
    levinson_durbin(scratch_r, order, coeffs_out);

    for (int i = 0; i < order; ++i) seeds_out[i] = (i < N) ? signal[i] : 0;
    for (int n = 0; n < N; ++n) residuals_out[n] = 0;
    for (int n = order; n < N; ++n) {
        double pred = 0.0;
        for (int k = 0; k < order; ++k) pred += coeffs_out[k] * scratch_x[n - k - 1];
        // compute residual as rounded difference to avoid double-rounding later
        residuals_out[n] = (int32_t)lround((double)signal[n] - pred);
    }
}

// ----------------------------- 
// reconstruct sample-by-sample (handles bypass)
// -----------------------------

// This function reconstructs sequentially using coefficients and earlier reconstructed samples.

static inline double predict_from_reconstructed(const double *coeffs, int order, const int16_t *recon, int n) {

    double pred = 0.0;
    for (int k = 0; k < order; ++k) pred += coeffs[k] * (double)recon[n - k - 1];
    return pred;

}

// ----------------------------- 
// heuristics 
// -----------------------------

static double mean_abs_res(const int32_t *res, int n) {

    double s = 0.0;
    for (int i = 0; i < n; ++i) s += fabs((double)res[i]);
    return (n > 0) ? s / n : 0.0;

}

static int compute_optimal_rice_k(const int32_t *res, int n) {

    double mu = mean_abs_res(res, n);
    if (mu < 0.5) return 0;
    int k = (int)floor(log2(mu + 1.0));
    if (k < 0) k = 0;
    if (k > 15) k = 15;
    return k;

}

static int select_optimal_order(const int16_t *signal, int N, int max_order) {

    if (N <= MIN_ORDER) return MIN_ORDER;
    double best_gain = 0.0; int best_order = MIN_ORDER;
    int sample_count = (N < 1024) ? N : 1024;

    for (int order = MIN_ORDER; order <= max_order; order += 2) {

        double *x = (double*)malloc(sizeof(double) * sample_count);
        double *r = (double*)calloc(order+1, sizeof(double));
        double *a = (double*)calloc(order, sizeof(double));

        if (!x || !r || !a) { free(x); free(r); free(a); return MIN_ORDER; }

        for (int i = 0; i < sample_count; ++i) x[i] = (double)signal[i];

        compute_autocorr_frame(x, sample_count, order, r);
        levinson_durbin(r, order, a);
        double mean = 0.0, var_res = 0.0, var_sig = 0.0;

        for (int i = 0; i < sample_count; ++i) mean += x[i]; mean /= sample_count;

        for (int n = 0; n < sample_count; ++n) {

            double pred = 0.0;
            
            if (n >= order) for (int k = 0; k < order; ++k) pred += a[k] * x[n - k - 1];
            
            double e = x[n] - pred; var_res += e * e;
            double d = x[n] - mean; var_sig += d * d;

        }

        var_res /= sample_count; var_sig /= sample_count;
        if (var_res <= 1e-12) var_res = 1e-12;
        if (var_sig <= 1e-12) var_sig = 1e-12;
        double gain = var_sig / var_res;
        if (gain > best_gain) { best_gain = gain; best_order = order; }
        free(x); free(r); free(a);
        if (order > MIN_ORDER && gain / best_gain < 1.05) break;

    }
    return best_order;

}

// ----------------------------- 
// Rice encode/decode 
// -----------------------------

static inline void rice_encode_to_bw(BitWriter *bw, int32_t signed_val, int k) {

    uint32_t u = ((uint32_t)signed_val << 1) ^ ((uint32_t)(signed_val >> 31));
    uint32_t q = u >> k;
    uint32_t r = u & ((1u << k) - 1u);
    for (uint32_t i = 0; i < q; ++i) bw_write_bit(bw, 1);
    bw_write_bit(bw, 0);
    if (k > 0) bw_write_bits(bw, r, k);

}

static inline int32_t rice_decode_from_br(BitReader *br, int k) {

    uint32_t q = 0;

    while (1) {

        int b = br_read_bit(br);
        if (b < 0) return INT32_MIN;
        if (b == 1) q++; else break;

    }

    uint32_t r = 0;
    if (k > 0) { uint32_t tmp; if (br_read_bits(br, k, &tmp) < 0) return INT32_MIN; r = tmp; }
    uint32_t u = (q << k) + r;
    int32_t decoded = (int32_t)((u >> 1) ^ (-(int32_t)(u & 1)));
    return decoded;

}

// ----------------------------- 
// Encoder (frame-based, with transient bypass and float32 coeffs) 
// -----------------------------

static void encode_file(const char *in_wav, const char *out_blos, int max_order) {

    SF_INFO sfinfo; memset(&sfinfo, 0, sizeof(sfinfo));

    SNDFILE *in = sf_open(in_wav, SFM_READ, &sfinfo);
    if (!in) { fprintf(stderr,"Failed to open '%s'\n", in_wav); exit(1); }

    int frames_total = (int)sfinfo.frames;
    int channels = sfinfo.channels;

    if (channels < 1 || channels > 2) { fprintf(stderr,"Only mono/stereo supported\n"); sf_close(in); exit(1); }

    printf("[Encode] %s: %d frames, %d channels, %d Hz\n", in_wav, frames_total, channels, sfinfo.samplerate);

    int total_samples = frames_total * channels;
    int16_t *buf = (int16_t*)malloc(sizeof(int16_t) * total_samples);
    if (!buf) { fprintf(stderr,"OOM\n"); sf_close(in); exit(1); }
    sf_readf_short(in, buf, frames_total);
    sf_close(in);

    int16_t *L = (int16_t*)malloc(sizeof(int16_t) * frames_total);
    int16_t *R = (int16_t*)malloc(sizeof(int16_t) * frames_total);
    if (channels == 1) for (int i = 0; i < frames_total; ++i) L[i] = R[i] = buf[i];
    else for (int i = 0; i < frames_total; ++i) { L[i] = buf[2*i]; R[i] = buf[2*i+1]; }
    free(buf);

    FILE *out = fopen(out_blos, "wb");

    if (!out) { fprintf(stderr,"Failed to open '%s'\n", out_blos); exit(1); }

    // global header
    fwrite(MAGIC, 1, 4, out);
    uint32_t sr = (uint32_t)sfinfo.samplerate; fwrite(&sr, sizeof(uint32_t), 1, out);
    uint32_t fr = (uint32_t)frames_total; fwrite(&fr, sizeof(uint32_t), 1, out);
    uint8_t ch = 2; fwrite(&ch, sizeof(uint8_t), 1, out);
    uint32_t frame_sz = FRAME_SIZE; fwrite(&frame_sz, sizeof(uint32_t), 1, out);
    uint32_t num_frames = (frames_total + FRAME_SIZE - 1) / FRAME_SIZE; fwrite(&num_frames, sizeof(uint32_t), 1, out);

    // reusable buffers
    int maxF = FRAME_SIZE;

    int16_t *Mframe = (int16_t*)malloc(sizeof(int16_t) * maxF);
    int16_t *Sframe = (int16_t*)malloc(sizeof(int16_t) * maxF);
    double *scratch_x = (double*)malloc(sizeof(double) * maxF);
    double *scratch_r = (double*)malloc(sizeof(double) * (MAX_ORDER + 1));
    double *coeffsA = (double*)malloc(sizeof(double) * MAX_ORDER);
    double *coeffsB = (double*)malloc(sizeof(double) * MAX_ORDER);
    double *fcoeffsA = (double*)malloc(sizeof(double) * MAX_ORDER);
    double *fcoeffsB = (double*)malloc(sizeof(double) * MAX_ORDER);
    int32_t *resA = (int32_t*)malloc(sizeof(int32_t) * maxF);
    int32_t *resB = (int32_t*)malloc(sizeof(int32_t) * maxF);
    int16_t *seedA = (int16_t*)malloc(sizeof(int16_t) * MAX_ORDER);
    int16_t *seedB = (int16_t*)malloc(sizeof(int16_t) * MAX_ORDER);

    BitWriter bw; bw_init(&bw, out);
    printf("[Encode] frames=%u frame_size=%u\n", num_frames, frame_sz);

    for (uint32_t f = 0; f < num_frames; ++f) {

        int start = f * FRAME_SIZE;
        int fsize = (start + FRAME_SIZE <= frames_total) ? FRAME_SIZE : (frames_total - start);

        // build M/S
        for (int i = 0; i < fsize; ++i) {

            int idx = start + i;
            int32_t l = L[idx], r = R[idx];
            Mframe[i] = (int16_t)((l + r) / 2);
            Sframe[i] = (int16_t)(l - r);

        }

        // pick order
        int orderM = select_optimal_order(Mframe, fsize, max_order);
        int orderS = select_optimal_order(Sframe, fsize, max_order);
        int order = (orderM > orderS) ? orderM : orderS;

        if (order < MIN_ORDER) order = MIN_ORDER;
        if (order > max_order) order = max_order;
        if (order >= fsize) order = (fsize > 1) ? (fsize - 1) : 1;

        // compute LPC & residuals
        compute_lpc_and_residuals_frame(Mframe, fsize, order, coeffsA, resA, seedA, scratch_x, scratch_r);
        compute_lpc_and_residuals_frame(Sframe, fsize, order, coeffsB, resB, seedB, scratch_x, scratch_r);

        // convert coeffs to double for storage (reduce quantization compared to float32)
        for (int i = 0; i < order; ++i) { fcoeffsA[i] = (double)coeffsA[i]; fcoeffsB[i] = (double)coeffsB[i]; }

        // choose Rice k
        int kA = (fsize > order) ? compute_optimal_rice_k(resA + order, fsize - order) : 0;
        int kB = (fsize > order) ? compute_optimal_rice_k(resB + order, fsize - order) : 0;

        // frame header (order(1), kA(1), kB(1), pad(1), then doubles size: order*8*2)
        uint32_t fsize32 = (uint32_t)fsize; fwrite(&fsize32, sizeof(uint32_t), 1, out);
        uint8_t order8 = (uint8_t)order; fwrite(&order8, sizeof(uint8_t), 1, out);
        uint8_t kA8 = (uint8_t)kA; fwrite(&kA8, sizeof(uint8_t), 1, out);
        uint8_t kB8 = (uint8_t)kB; fwrite(&kB8, sizeof(uint8_t), 1, out);

        // write seeds
        fwrite(seedA, sizeof(int16_t), order, out);
        fwrite(seedB, sizeof(int16_t), order, out);

        // write double coeffs directly (8 bytes each) to reduce quantization
        fwrite(fcoeffsA, sizeof(double), order, out);
        fwrite(fcoeffsB, sizeof(double), order, out);

        // now write residuals with per-sample escape:
        // For each n >= order:
        //   if abs(residual) > ESCAPE_THRESHOLD -> write bit 1 then write 16-bit raw sample (Mframe[n])
        //   else write bit 0 then Rice-code residual
        int escape_countA = 0;
        int escape_countB = 0;
        for (int i = order; i < fsize; ++i) {

            int32_t rsv = resA[i];

            if (llabs((long long)rsv) > ESCAPE_THRESHOLD) {

                bw_write_bit(&bw, 1);
                // write raw 16-bit sample value
                uint16_t raw = (uint16_t)( (uint16_t)Mframe[i] );
                bw_write_bits(&bw, raw, 16);
                escape_countA++;

            } else {

                bw_write_bit(&bw, 0);
                rice_encode_to_bw(&bw, rsv, kA);

            }

        }

        for (int i = order; i < fsize; ++i) {

            int32_t rsv = resB[i];
            if (llabs((long long)rsv) > ESCAPE_THRESHOLD) {

                bw_write_bit(&bw, 1);
                uint16_t raw = (uint16_t)( (uint16_t)Sframe[i] );
                bw_write_bits(&bw, raw, 16);
                escape_countB++;

            } else {

                bw_write_bit(&bw, 0);
                rice_encode_to_bw(&bw, rsv, kB);

            }

        }

        bw_flush(&bw); // align to byte boundary

        double mean_absA = mean_abs_res(resA + order, fsize - order);
        double mean_absB = mean_abs_res(resB + order, fsize - order);
        printf("Frame %u/%u: order=%d kA=%d kB=%d meanA=%.1f meanB=%.1f escA=%d escB=%d\n",
               f+1, num_frames, order, kA, kB, mean_absA, mean_absB, escape_countA, escape_countB);

        if ((f & (PROGRESS_STEP - 1)) == 0) printf("Encoded frame %u/%u\n", f+1, num_frames);

    }

    bw_flush(&bw);
    fclose(out);

    // cleanup
    free(Mframe); free(Sframe); free(scratch_x); free(scratch_r);
    free(coeffsA); free(coeffsB); free(fcoeffsA); free(fcoeffsB);
    free(resA); free(resB); free(seedA); free(seedB);
    free(L); free(R);
    printf("[Encode] done: %s\n", out_blos);

}

// ----------------------------- 
// Decoder (handles escapes and float32 coeffs)
// ----------------------------- 

static void decode_file(const char *in_blos, const char *out_wav) {

    FILE *in = fopen(in_blos, "rb");
    
    if (!in) { fprintf(stderr,"Failed to open %s\n", in_blos); exit(1); }
    char magic[4]; if (fread(magic,1,4,in) != 4) { fprintf(stderr,"read error\n"); exit(1); }

    if (memcmp(magic, MAGIC, 4) != 0) { fprintf(stderr,"Not a BLOS file\n"); exit(1); }

    uint32_t sr; fread(&sr, sizeof(uint32_t), 1, in);
    uint32_t frames_total; fread(&frames_total, sizeof(uint32_t), 1, in);
    uint8_t channels; fread(&channels, sizeof(uint8_t), 1, in);
    uint32_t frame_sz; fread(&frame_sz, sizeof(uint32_t), 1, in);
    uint32_t num_frames; fread(&num_frames, sizeof(uint32_t), 1, in);

    printf("[Decode] frames_total=%u sr=%u frame_sz=%u num_frames=%u\n", frames_total, sr, frame_sz, num_frames);

    SF_INFO sfinfo; memset(&sfinfo,0,sizeof(sfinfo));
    sfinfo.channels = 2; sfinfo.samplerate = (int)sr; sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
    SNDFILE *out = sf_open(out_wav, SFM_WRITE, &sfinfo);
    
    if (!out) { fprintf(stderr,"Failed to open output %s\n", out_wav); exit(1); }

    BitReader br; br_init(&br, in);

    int maxF = frame_sz;
    
    double *coeffsA = (double*)malloc(sizeof(double) * MAX_ORDER);
    double *coeffsB = (double*)malloc(sizeof(double) * MAX_ORDER);
    int32_t *resA = (int32_t*)malloc(sizeof(int32_t) * maxF);
    int32_t *resB = (int32_t*)malloc(sizeof(int32_t) * maxF);
    int16_t *seedA = (int16_t*)malloc(sizeof(int16_t) * MAX_ORDER);
    int16_t *seedB = (int16_t*)malloc(sizeof(int16_t) * MAX_ORDER);
    double *fcoeffsA = (double*)malloc(sizeof(double) * MAX_ORDER);
    double *fcoeffsB = (double*)malloc(sizeof(double) * MAX_ORDER);
    int16_t *Mframe = (int16_t*)malloc(sizeof(int16_t) * maxF);
    int16_t *Sframe = (int16_t*)malloc(sizeof(int16_t) * maxF);

    for (uint32_t f = 0; f < num_frames; ++f) {
    
        uint32_t fsize; if (fread(&fsize, sizeof(uint32_t), 1, in) != 1) { fprintf(stderr,"header read error\n"); exit(1); }
        uint8_t order8; fread(&order8, sizeof(uint8_t), 1, in);
        uint8_t kA8; fread(&kA8, sizeof(uint8_t), 1, in);
        uint8_t kB8; fread(&kB8, sizeof(uint8_t), 1, in);
        int order = (int)order8, kA = (int)kA8, kB = (int)kB8;

        if (order <= 0 || order > MAX_ORDER) { fprintf(stderr,"invalid order %d\n", order); exit(1); }

        if (fread(seedA, sizeof(int16_t), order, in) != (size_t)order) { fprintf(stderr,"seed read error\n"); exit(1); }
        if (fread(seedB, sizeof(int16_t), order, in) != (size_t)order) { fprintf(stderr,"seed read error\n"); exit(1); }
        if (fread(fcoeffsA, sizeof(double), order, in) != (size_t)order) { fprintf(stderr,"coeff read error\n"); exit(1); }
        if (fread(fcoeffsB, sizeof(double), order, in) != (size_t)order) { fprintf(stderr,"coeff read error\n"); exit(1); }

        // convert stored doubles back to prediction coeffs
        for (int i = 0; i < order; ++i) { coeffsA[i] = (double)fcoeffsA[i]; coeffsB[i] = (double)fcoeffsB[i]; }

        // initialize reconstructed seeds
        for (int i = 0; i < order && i < (int)fsize; ++i) { Mframe[i] = seedA[i]; Sframe[i] = seedB[i]; }

        // decode residuals with escapes and reconstruct on the fly (so predictions use reconstructed samples)
        for (int n = order; n < (int)fsize; ++n) {
    
            int flag = br_read_bit(&br);
            if (flag < 0) { fprintf(stderr,"bitstream error\n"); exit(1); }
            if (flag == 1) {
    
                uint32_t tmp;
                if (br_read_bits(&br, 16, &tmp) < 0) { fprintf(stderr,"read raw sample error\n"); exit(1); }
                int16_t sample = (int16_t)tmp;
                Mframe[n] = sample;
    
            } else {
    
                int32_t decoded = rice_decode_from_br(&br, kA);
                if (decoded == INT32_MIN) { fprintf(stderr,"decode error M\n"); exit(1); }
                double pred = predict_from_reconstructed(coeffsA, order, Mframe, n);
                int32_t s = (int32_t)lround(pred + (double)decoded);
                if (s > 32767) s = 32767; if (s < -32768) s = -32768;
                Mframe[n] = (int16_t)s;
    
            }
    
        }

        for (int n = order; n < (int)fsize; ++n) {
    
            int flag = br_read_bit(&br);
            if (flag < 0) { fprintf(stderr,"bitstream error\n"); exit(1); }
            if (flag == 1) {
    
                uint32_t tmp;
                if (br_read_bits(&br, 16, &tmp) < 0) { fprintf(stderr,"read raw sample error\n"); exit(1); }
                int16_t sample = (int16_t)tmp;
                Sframe[n] = sample;
    
            } else {
    
                int32_t decoded = rice_decode_from_br(&br, kB);
                if (decoded == INT32_MIN) { fprintf(stderr,"decode error S\n"); exit(1); }
                double pred = predict_from_reconstructed(coeffsB, order, Sframe, n);
                int32_t s = (int32_t)lround(pred + (double)decoded);
                if (s > 32767) s = 32767; if (s < -32768) s = -32768;
                Sframe[n] = (int16_t)s;
    
            }
    
        }

        // ensure header alignment for next frame
        br_byte_align(&br);

        // convert M/S -> L/R and write
        int16_t *outbuf = (int16_t*)malloc(sizeof(int16_t) * fsize * 2);
        for (int i = 0; i < (int)fsize; ++i) {
    
            int32_t m = Mframe[i], s = Sframe[i];
            int32_t l = m + (s / 2);
            int32_t r = m - (s / 2);
            if (l > 32767) l = 32767; if (l < -32768) l = -32768;
            if (r > 32767) r = 32767; if (r < -32768) r = -32768;
            outbuf[2*i] = (int16_t)l; outbuf[2*i+1] = (int16_t)r;
    
        }
        sf_writef_short(out, outbuf, fsize);
        free(outbuf);

        if ((f & (PROGRESS_STEP - 1)) == 0) printf("Decoded frame %u/%u\n", f+1, num_frames);
    
    }

    sf_close(out);
    fclose(in);

    free(coeffsA); free(coeffsB); free(resA); free(resB);
    free(seedA); free(seedB); free(fcoeffsA); free(fcoeffsB);
    free(Mframe); free(Sframe);

    printf("[Decode] wrote %s\n", out_wav);

}

// ----------------------------- 
// CLI
// -----------------------------

int main(int argc, char **argv) {

    if (argc == 4 && strcmp(argv[1], "-d") == 0) {

        decode_file(argv[2], argv[3]);

    } else if (argc == 3) {

        encode_file(argv[1], argv[2], MAX_ORDER);

    } else {

        printf("Usage:\n  Encode: %s input.wav output.blos\n  Decode: %s -d input.blos output.wav\n", argv[0], argv[0]);
        return 1;

    }
    return 0;

}
