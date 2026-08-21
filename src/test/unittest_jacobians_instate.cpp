#include <gtest/gtest.h>
//#include <unsupported/Eigen/MatrixFunctions>

#define private public

#include "alias.h"
#include "mm.h"
#include "group.h"
#include "graph.h"

#include "unittest_helpers.h"

#include "feature.h"


//using namespace Eigen;
using namespace xivo;

/* Class to check dxc(now)_d[everything else]. We wil */
class InstateJacobiansTest : public ::testing::Test {
  protected:
    void SetUp() override {

        // Create feature object and testing parameters
        MemoryManager::Create(256, 128);
        auto cfg_ = LoadJson("src/test/camera_configs.json");
        Camera::Create(cfg_["perfect_pinhole"]);
        delta = 1e-6;
        tol = 9e-4;

        // IMU measurement
        gyro = Vec3::Random();


        // Set nominal and error variables to random values
        std::default_random_engine generator;
        gsbr_nom = SE3::sampleUniform(generator);
        gsb_nom = SE3::sampleUniform(generator);
        gbc_nom = SE3::sampleUniform(generator);
        Cg_nom = Mat3::Identity();
        bg_nom << 0, 0, 0;
        td_nom = 0.005;
        Vsb_nom = Vec3::Random();

        // Initialize the error variables
        Wsbr_err = Vec3::Zero();
        Tsbr_err = Vec3::Zero();
        Wsb_err = Vec3::Zero();
        Tsb_err = Vec3::Zero();
        Wbc_err = Vec3::Zero();
        Tbc_err = Vec3::Zero();
        Cg_err = Mat3::Zero();
        bg_err = Vec3::Zero();
        td_err = 0.0;

        // Set reference Rr and Tr for the feature
        Vec2 xp(25, 46);
        f = Feature::Create(xp(0), xp(1)); // sets x_(2) = 2.0 = log(z) (or 1/z)

        // Compute coordinates of internal state
        Vec2 xc = Camera::instance()->UnProject(xp);
        f->x_(0) = xc(0);
        f->x_(1) = xc(1);
        group = Group::Create(gsbr_nom.so3(), gsbr_nom.translation());
        group->SetSind(0);
        f->ref_ = group;
        f->SetSind(0);

        // Compute the analytic Jacobians and nominal states
        ComputeNominalStates();
        f->ComputeJacobian(gsb_nom.so3().matrix(), gsb_nom.translation(),
                           gbc_nom.so3().matrix(), gbc_nom.translation(),
                           gyro, Cg_nom, bg_nom, Vsb_nom, td_nom);
    }

    Vec3 ComputeXcn() {
        SO3 Rsbr_err = SO3::exp(Wsbr_err);

        SO3 Rsbr = gsbr_nom.so3() * Rsbr_err;
        Vec3 Tsbr = gsbr_nom.translation() + Tsbr_err;
        
        Cg = Cg_nom + Cg_err;
        bg = bg_nom + bg_err;

#ifdef USE_ONLINE_TEMPORAL_CALIB
        Vec3 angvel_nom = Cg*gyro - bg;
        Vec3 angvel_err = Cg_err*gyro - bg_err;
        SO3 delta_rot = SO3::exp(angvel_nom*td_err + angvel_err*(td_nom + td_err));
        Rsb = gsb_nom.so3() * SO3::exp(Wsb_err) * delta_rot;
        Tsb = gsb_nom.translation() + Tsb_err + Vsb_nom*td_err;
#else
        Rsb = gsb_nom.so3() * SO3::exp(Wsb_err);
        Tsb = gsb_nom.translation() + Tsb_err;
#endif

        Rbc = gbc_nom.so3() * SO3::exp(Wbc_err);
        Tbc = gbc_nom.translation() + Tbc_err;
        td = td_nom + td_err;


        Vec3 Xc = f->Xc();
        Vec3 Xs = Rsbr*(Rbc*Xc + Tbc) + Tsbr;

        SE3 gcb = SE3(Rbc, Tbc).inverse();
        SE3 gbs = SE3(Rsb, Tsb).inverse();
        Vec3 Xcn = gcb * gbs * Xs;

        return Xcn;
    }

    void ComputeNominalStates() {
        Xc_nom = f->Xc(nullptr);
        Xs_nom = gsbr_nom * gbc_nom * Xc_nom;
        Xcn_nom = gbc_nom.inverse() * gsb_nom.inverse() * Xs_nom;
    }

    // Feature Object and Memory Manager
    MemoryManagerPtr mm;
    CameraPtr cam;
    GroupPtr group;
    FeaturePtr f;

    // Fake IMU measurement
    Vec3 gyro;

    // numerical tolerance
    number_t tol;

    // Real values (= nominal + error)
    SO3 Rsbr;
    Vec3 Tsbr;
    SO3 Rsb;
    Vec3 Tsb;
    SO3 Rbc;
    Vec3 Tbc;
    Mat3 Cg;
    Vec3 bg;
    number_t td;

    // Nominal state variables containing placeholder values
    SE3 gsbr_nom;
    SE3 gsb_nom;
    SE3 gbc_nom;
    Mat3 Cg_nom;
    Vec3 bg_nom;
    number_t td_nom;
    Vec3 Vsb_nom;

    // Error variables containing placeholder values
    Vec3 Wsbr_err;
    Vec3 Tsbr_err;
    Vec3 Wsb_err;
    Vec3 Tsb_err;
    Vec3 Wbc_err;
    Vec3 Tbc_err;
    Mat3 Cg_err;
    Vec3 bg_err;
    number_t td_err;

    // finite difference
    number_t delta;

    // Values to save for debugging
    Vec3 Xc_nom;
    Vec3 Xs_nom;
    Vec3 Xcn_nom;
};


TEST_F(InstateJacobiansTest, NominalStates) {
    EXPECT_NEAR(Xc_nom(0), f->cache_.Xc(0), tol);
    EXPECT_NEAR(Xc_nom(1), f->cache_.Xc(1), tol);
    EXPECT_NEAR(Xc_nom(2), f->cache_.Xc(2), tol);

    EXPECT_NEAR(Xs_nom(0), f->cache_.Xs(0), tol);
    EXPECT_NEAR(Xs_nom(1), f->cache_.Xs(1), tol);
    EXPECT_NEAR(Xs_nom(2), f->cache_.Xs(2), tol);

    EXPECT_NEAR(Xcn_nom(0), f->cache_.Xcn(0), tol);
    EXPECT_NEAR(Xcn_nom(1), f->cache_.Xcn(1), tol);
    EXPECT_NEAR(Xcn_nom(2), f->cache_.Xcn(2), tol);
}


TEST_F(InstateJacobiansTest, Wsbr) {
    Vec3 Xcn0 = ComputeXcn();

    Wsbr_err(0) = delta;
    Vec3 Xcn1_0 = ComputeXcn();
    Wsbr_err(0) = 0;

    Wsbr_err(1) = delta;
    Vec3 Xcn1_1 = ComputeXcn();
    Wsbr_err(1) = 0;

    Wsbr_err(2) = delta;
    Vec3 Xcn1_2 = ComputeXcn();
    Wsbr_err(2) = 0;

    Vec3 dXcn_dWsbr0 = (Xcn1_0 - Xcn0) / delta;
    Vec3 dXcn_dWsbr1 = (Xcn1_1 - Xcn0) / delta;
    Vec3 dXcn_dWsbr2 = (Xcn1_2 - Xcn0) / delta;

    EXPECT_NEAR(dXcn_dWsbr0(0), f->cache_.dXcn_dWsbr(0,0), tol);
    EXPECT_NEAR(dXcn_dWsbr0(1), f->cache_.dXcn_dWsbr(1,0), tol);
    EXPECT_NEAR(dXcn_dWsbr0(2), f->cache_.dXcn_dWsbr(2,0), tol);
    
    EXPECT_NEAR(dXcn_dWsbr1(0), f->cache_.dXcn_dWsbr(0,1), tol);
    EXPECT_NEAR(dXcn_dWsbr1(1), f->cache_.dXcn_dWsbr(1,1), tol);
    EXPECT_NEAR(dXcn_dWsbr1(2), f->cache_.dXcn_dWsbr(2,1), tol);

    EXPECT_NEAR(dXcn_dWsbr2(0), f->cache_.dXcn_dWsbr(0,2), tol);
    EXPECT_NEAR(dXcn_dWsbr2(1), f->cache_.dXcn_dWsbr(1,2), tol);
    EXPECT_NEAR(dXcn_dWsbr2(2), f->cache_.dXcn_dWsbr(2,2), tol);
}


TEST_F(InstateJacobiansTest, Tr) {
    Vec3 Xcn0 = ComputeXcn();

    Tsbr_err(0) = delta;
    Vec3 Xcn1_0 = ComputeXcn();
    Tsbr_err(0) = 0;

    Tsbr_err(1) = delta;
    Vec3 Xcn1_1 = ComputeXcn();
    Tsbr_err(1) = 0;

    Tsbr_err(2) = delta;
    Vec3 Xcn1_2 = ComputeXcn();
    Tsbr_err(2) = 0;

    Vec3 dXcn_dTsbr0 = (Xcn1_0 - Xcn0) / delta;
    Vec3 dXcn_dTsbr1 = (Xcn1_1 - Xcn0) / delta;
    Vec3 dXcn_dTsbr2 = (Xcn1_2 - Xcn0) / delta;

    EXPECT_NEAR(dXcn_dTsbr0(0), f->cache_.dXcn_dTsbr(0,0), tol);
    EXPECT_NEAR(dXcn_dTsbr0(1), f->cache_.dXcn_dTsbr(1,0), tol);
    EXPECT_NEAR(dXcn_dTsbr0(2), f->cache_.dXcn_dTsbr(2,0), tol);
    
    EXPECT_NEAR(dXcn_dTsbr1(0), f->cache_.dXcn_dTsbr(0,1), tol);
    EXPECT_NEAR(dXcn_dTsbr1(1), f->cache_.dXcn_dTsbr(1,1), tol);
    EXPECT_NEAR(dXcn_dTsbr1(2), f->cache_.dXcn_dTsbr(2,1), tol);

    EXPECT_NEAR(dXcn_dTsbr2(0), f->cache_.dXcn_dTsbr(0,2), tol);
    EXPECT_NEAR(dXcn_dTsbr2(1), f->cache_.dXcn_dTsbr(1,2), tol);
    EXPECT_NEAR(dXcn_dTsbr2(2), f->cache_.dXcn_dTsbr(2,2), tol);
}


TEST_F(InstateJacobiansTest, Wsb) {
    Vec3 Xcn0 = ComputeXcn();

    Wsb_err(0) = delta;
    Vec3 Xcn1_0 = ComputeXcn();
    Wsb_err(0) = 0;

    Wsb_err(1) = delta;
    Vec3 Xcn1_1 = ComputeXcn();
    Wsb_err(1) = 0;

    Wsb_err(2) = delta;
    Vec3 Xcn1_2 = ComputeXcn();
    Wsb_err(2) = 0;

    Vec3 dXcn_dWsb0 = (Xcn1_0 - Xcn0) / delta;
    Vec3 dXcn_dWsb1 = (Xcn1_1 - Xcn0) / delta;
    Vec3 dXcn_dWsb2 = (Xcn1_2 - Xcn0) / delta;

    EXPECT_NEAR(dXcn_dWsb0(0), f->cache_.dXcn_dWsb(0,0), tol);
    EXPECT_NEAR(dXcn_dWsb0(1), f->cache_.dXcn_dWsb(1,0), tol);
    EXPECT_NEAR(dXcn_dWsb0(2), f->cache_.dXcn_dWsb(2,0), tol);
    
    EXPECT_NEAR(dXcn_dWsb1(0), f->cache_.dXcn_dWsb(0,1), tol);
    EXPECT_NEAR(dXcn_dWsb1(1), f->cache_.dXcn_dWsb(1,1), tol);
    EXPECT_NEAR(dXcn_dWsb1(2), f->cache_.dXcn_dWsb(2,1), tol);

    EXPECT_NEAR(dXcn_dWsb2(0), f->cache_.dXcn_dWsb(0,2), tol);
    EXPECT_NEAR(dXcn_dWsb2(1), f->cache_.dXcn_dWsb(1,2), tol);
    EXPECT_NEAR(dXcn_dWsb2(2), f->cache_.dXcn_dWsb(2,2), tol);
}


TEST_F(InstateJacobiansTest, Tsb) {
    Vec3 Xcn0 = ComputeXcn();

    Tsb_err(0) = delta;
    Vec3 Xcn1_0 = ComputeXcn();
    Tsb_err(0) = 0;

    Tsb_err(1) = delta;
    Vec3 Xcn1_1 = ComputeXcn();
    Tsb_err(1) = 0;

    Tsb_err(2) = delta;
    Vec3 Xcn1_2 = ComputeXcn();
    Tsb_err(2) = 0;

    Vec3 dXcn_dTsb0 = (Xcn1_0 - Xcn0) / delta;
    Vec3 dXcn_dTsb1 = (Xcn1_1 - Xcn0) / delta;
    Vec3 dXcn_dTsb2 = (Xcn1_2 - Xcn0) / delta;

    EXPECT_NEAR(dXcn_dTsb0(0), f->cache_.dXcn_dTsb(0,0), tol);
    EXPECT_NEAR(dXcn_dTsb0(1), f->cache_.dXcn_dTsb(1,0), tol);
    EXPECT_NEAR(dXcn_dTsb0(2), f->cache_.dXcn_dTsb(2,0), tol);
    
    EXPECT_NEAR(dXcn_dTsb1(0), f->cache_.dXcn_dTsb(0,1), tol);
    EXPECT_NEAR(dXcn_dTsb1(1), f->cache_.dXcn_dTsb(1,1), tol);
    EXPECT_NEAR(dXcn_dTsb1(2), f->cache_.dXcn_dTsb(2,1), tol);

    EXPECT_NEAR(dXcn_dTsb2(0), f->cache_.dXcn_dTsb(0,2), tol);
    EXPECT_NEAR(dXcn_dTsb2(1), f->cache_.dXcn_dTsb(1,2), tol);
    EXPECT_NEAR(dXcn_dTsb2(2), f->cache_.dXcn_dTsb(2,2), tol);
}


TEST_F(InstateJacobiansTest, Wbc) {
    Vec3 Xcn0 = ComputeXcn();

    Wbc_err(0) = delta;
    Vec3 Xcn1_0 = ComputeXcn();
    Wbc_err(0) = 0;

    Wbc_err(1) = delta;
    Vec3 Xcn1_1 = ComputeXcn();
    Wbc_err(1) = 0;

    Wbc_err(2) = delta;
    Vec3 Xcn1_2 = ComputeXcn();
    Wbc_err(2) = 0;

    Vec3 dXcn_dWbc0 = (Xcn1_0 - Xcn0) / delta;
    Vec3 dXcn_dWbc1 = (Xcn1_1 - Xcn0) / delta;
    Vec3 dXcn_dWbc2 = (Xcn1_2 - Xcn0) / delta;

    EXPECT_NEAR(dXcn_dWbc0(0), f->cache_.dXcn_dWbc(0,0), tol);
    EXPECT_NEAR(dXcn_dWbc0(1), f->cache_.dXcn_dWbc(1,0), tol);
    EXPECT_NEAR(dXcn_dWbc0(2), f->cache_.dXcn_dWbc(2,0), tol);
    
    EXPECT_NEAR(dXcn_dWbc1(0), f->cache_.dXcn_dWbc(0,1), tol);
    EXPECT_NEAR(dXcn_dWbc1(1), f->cache_.dXcn_dWbc(1,1), tol);
    EXPECT_NEAR(dXcn_dWbc1(2), f->cache_.dXcn_dWbc(2,1), tol);

    EXPECT_NEAR(dXcn_dWbc2(0), f->cache_.dXcn_dWbc(0,2), tol);
    EXPECT_NEAR(dXcn_dWbc2(1), f->cache_.dXcn_dWbc(1,2), tol);
    EXPECT_NEAR(dXcn_dWbc2(2), f->cache_.dXcn_dWbc(2,2), tol);
}


TEST_F(InstateJacobiansTest, Tbc) {
    Vec3 Xcn0 = ComputeXcn();

    Tbc_err(0) = delta;
    Vec3 Xcn1_0 = ComputeXcn();
    Tbc_err(0) = 0;

    Tbc_err(1) = delta;
    Vec3 Xcn1_1 = ComputeXcn();
    Tbc_err(1) = 0;

    Tbc_err(2) = delta;
    Vec3 Xcn1_2 = ComputeXcn();
    Tbc_err(2) = 0;

    Vec3 dXcn_dTbc0 = (Xcn1_0 - Xcn0) / delta;
    Vec3 dXcn_dTbc1 = (Xcn1_1 - Xcn0) / delta;
    Vec3 dXcn_dTbc2 = (Xcn1_2 - Xcn0) / delta;

    EXPECT_NEAR(dXcn_dTbc0(0), f->cache_.dXcn_dTbc(0,0), tol);
    EXPECT_NEAR(dXcn_dTbc0(1), f->cache_.dXcn_dTbc(1,0), tol);
    EXPECT_NEAR(dXcn_dTbc0(2), f->cache_.dXcn_dTbc(2,0), tol);
    
    EXPECT_NEAR(dXcn_dTbc1(0), f->cache_.dXcn_dTbc(0,1), tol);
    EXPECT_NEAR(dXcn_dTbc1(1), f->cache_.dXcn_dTbc(1,1), tol);
    EXPECT_NEAR(dXcn_dTbc1(2), f->cache_.dXcn_dTbc(2,1), tol);

    EXPECT_NEAR(dXcn_dTbc2(0), f->cache_.dXcn_dTbc(0,2), tol);
    EXPECT_NEAR(dXcn_dTbc2(1), f->cache_.dXcn_dTbc(1,2), tol);
    EXPECT_NEAR(dXcn_dTbc2(2), f->cache_.dXcn_dTbc(2,2), tol);
}





#ifdef USE_ONLINE_TEMPORAL_CALIB
TEST_F(InstateJacobiansTest, td) {
    Vec3 Xcn0 = ComputeXcn();
    td_err = delta;

    Vec3 Xcn1 = ComputeXcn();

    Vec3 dXcn_dtd = (Xcn1 - Xcn0) / delta;
    EXPECT_NEAR(dXcn_dtd(0), f->cache_.dXcn_dtd(0), tol);
    EXPECT_NEAR(dXcn_dtd(1), f->cache_.dXcn_dtd(1), tol);
    EXPECT_NEAR(dXcn_dtd(2), f->cache_.dXcn_dtd(2), tol);
}


#ifdef USE_ONLINE_IMU_CALIB
TEST_F(InstateJacobiansTest, bg) {
    Vec3 Xcn0 = ComputeXcn();

    bg_err(0) = delta;
    Vec3 Xcn1_0 = ComputeXcn();
    bg_err(0) = 0;

    bg_err(1) = delta;
    Vec3 Xcn1_1 = ComputeXcn();
    bg_err(1) = 0;

    bg_err(2) = delta;
    Vec3 Xcn1_2 = ComputeXcn();
    bg_err(2) = 0;

    Vec3 dXcn_dbg0 = (Xcn1_0 - Xcn0) / delta;
    Vec3 dXcn_dbg1 = (Xcn1_1 - Xcn0) / delta;
    Vec3 dXcn_dbg2 = (Xcn1_2 - Xcn0) / delta;

    EXPECT_NEAR(dXcn_dbg0(0), f->cache_.dXcn_dbg(0,0), tol);
    EXPECT_NEAR(dXcn_dbg0(1), f->cache_.dXcn_dbg(1,0), tol);
    EXPECT_NEAR(dXcn_dbg0(2), f->cache_.dXcn_dbg(2,0), tol);
    
    EXPECT_NEAR(dXcn_dbg1(0), f->cache_.dXcn_dbg(0,1), tol);
    EXPECT_NEAR(dXcn_dbg1(1), f->cache_.dXcn_dbg(1,1), tol);
    EXPECT_NEAR(dXcn_dbg1(2), f->cache_.dXcn_dbg(2,1), tol);

    EXPECT_NEAR(dXcn_dbg2(0), f->cache_.dXcn_dbg(0,2), tol);
    EXPECT_NEAR(dXcn_dbg2(1), f->cache_.dXcn_dbg(1,2), tol);
    EXPECT_NEAR(dXcn_dbg2(2), f->cache_.dXcn_dbg(2,2), tol);
}


TEST_F(InstateJacobiansTest, Cg) {
    Vec3 Xcn0 = ComputeXcn();

    Cg_err(0,0) = delta;
    Vec3 Xcn1_00 = ComputeXcn();
    Cg_err(0,0) = 0;
    Vec3 dXcn_dCg00 = (Xcn1_00 - Xcn0) / delta;
    EXPECT_NEAR(dXcn_dCg00(0), f->cache_.dXcn_dCg(0,0), tol);
    EXPECT_NEAR(dXcn_dCg00(1), f->cache_.dXcn_dCg(1,0), tol);
    EXPECT_NEAR(dXcn_dCg00(2), f->cache_.dXcn_dCg(2,0), tol);

    Cg_err(0,1) = delta;
    Vec3 Xcn1_01 = ComputeXcn();
    Cg_err(0,1) = 0;
    Vec3 dXcn_dCg01 = (Xcn1_01 - Xcn0) / delta;
    EXPECT_NEAR(dXcn_dCg01(0), f->cache_.dXcn_dCg(0,1), tol);
    EXPECT_NEAR(dXcn_dCg01(1), f->cache_.dXcn_dCg(1,1), tol);
    EXPECT_NEAR(dXcn_dCg01(2), f->cache_.dXcn_dCg(2,1), tol);

    Cg_err(0,2) = delta;
    Vec3 Xcn1_02 = ComputeXcn();
    Cg_err(0,2) = 0;
    Vec3 dXcn_dCg02 = (Xcn1_02 - Xcn0) / delta;
    EXPECT_NEAR(dXcn_dCg02(0), f->cache_.dXcn_dCg(0,2), tol);
    EXPECT_NEAR(dXcn_dCg02(1), f->cache_.dXcn_dCg(1,2), tol);
    EXPECT_NEAR(dXcn_dCg02(2), f->cache_.dXcn_dCg(2,2), tol);

    Cg_err(1,0) = delta;
    Vec3 Xcn1_10 = ComputeXcn();
    Cg_err(1,0) = 0;
    Vec3 dXcn_dCg10 = (Xcn1_10 - Xcn0) / delta;
    EXPECT_NEAR(dXcn_dCg10(0), f->cache_.dXcn_dCg(0,3), tol);
    EXPECT_NEAR(dXcn_dCg10(1), f->cache_.dXcn_dCg(1,3), tol);
    EXPECT_NEAR(dXcn_dCg10(2), f->cache_.dXcn_dCg(2,3), tol);

    Cg_err(1,1) = delta;
    Vec3 Xcn1_11 = ComputeXcn();
    Cg_err(1,1) = 0;
    Vec3 dXcn_dCg11 = (Xcn1_11 - Xcn0) / delta;
    EXPECT_NEAR(dXcn_dCg11(0), f->cache_.dXcn_dCg(0,4), tol);
    EXPECT_NEAR(dXcn_dCg11(1), f->cache_.dXcn_dCg(1,4), tol);
    EXPECT_NEAR(dXcn_dCg11(2), f->cache_.dXcn_dCg(2,4), tol);

    Cg_err(1,2) = delta;
    Vec3 Xcn1_12 = ComputeXcn();
    Cg_err(1,2) = 0;
    Vec3 dXcn_dCg12 = (Xcn1_12 - Xcn0) / delta;
    EXPECT_NEAR(dXcn_dCg12(0), f->cache_.dXcn_dCg(0,5), tol);
    EXPECT_NEAR(dXcn_dCg12(1), f->cache_.dXcn_dCg(1,5), tol);
    EXPECT_NEAR(dXcn_dCg12(2), f->cache_.dXcn_dCg(2,5), tol);

    Cg_err(2,0) = delta;
    Vec3 Xcn1_20 = ComputeXcn();
    Cg_err(2,0) = 0;
    Vec3 dXcn_dCg20 = (Xcn1_20 - Xcn0) / delta;
    EXPECT_NEAR(dXcn_dCg20(0), f->cache_.dXcn_dCg(0,6), tol);
    EXPECT_NEAR(dXcn_dCg20(1), f->cache_.dXcn_dCg(1,6), tol);
    EXPECT_NEAR(dXcn_dCg20(2), f->cache_.dXcn_dCg(2,6), tol);

    Cg_err(2,1) = delta;
    Vec3 Xcn1_21 = ComputeXcn();
    Cg_err(2,1) = 0;
    Vec3 dXcn_dCg21 = (Xcn1_21 - Xcn0) / delta;
    EXPECT_NEAR(dXcn_dCg21(0), f->cache_.dXcn_dCg(0,7), tol);
    EXPECT_NEAR(dXcn_dCg21(1), f->cache_.dXcn_dCg(1,7), tol);
    EXPECT_NEAR(dXcn_dCg21(2), f->cache_.dXcn_dCg(2,7), tol);

    Cg_err(2,2) = delta;
    Vec3 Xcn1_22 = ComputeXcn();
    Cg_err(2,2) = 0;
    Vec3 dXcn_dCg22 = (Xcn1_22 - Xcn0) / delta;
    EXPECT_NEAR(dXcn_dCg22(0), f->cache_.dXcn_dCg(0,8), tol);
    EXPECT_NEAR(dXcn_dCg22(1), f->cache_.dXcn_dCg(1,8), tol);
    EXPECT_NEAR(dXcn_dCg22(2), f->cache_.dXcn_dCg(2,8), tol);

}
#endif
#endif
/** Regression test for `Feature::FillJacobianBlock`.
 *
 * `ComputeJacobian` fills the 2 x kFullSize row block `J_`; `FillJacobianBlock`
 * copies the non-zero blocks of `J_` into the stacked measurement Jacobian `H`
 * used by `Estimator::FilterUpdate`. Every block must land at the *same* column
 * offset it occupies in `J_`.
 *
 * This used to be false for the reference group: the rotation block and the
 * translation block were both written to `goff`, so the rotation columns held
 * the translation Jacobian and the translation columns stayed zero. The
 * existing tests all check `J_` (or `cache_`), never the copy, so nothing
 * caught it.
 */
TEST_F(InstateJacobiansTest, FillJacobianBlockCopiesEveryBlock) {
    const int offset = 4;  // a non-zero row offset, as in a real stacked update
    MatX H(offset + 2, kFullSize);
    H.setZero();

    f->FillJacobianBlock(H, offset);

    const int goff = kGroupBegin + 6 * f->ref_->sind();
    const int foff = kFeatureBegin + 3 * f->sind();

    // Compare a 2x3 block of H against the same columns of J_. Wrapped in a
    // lambda because the commas in block<2, 3> would be read as macro argument
    // separators inside EXPECT_*.
    auto block_matches = [&](int col) {
      return H.block(offset, col, 2, 3).isApprox(f->J_.block(0, col, 2, 3));
    };

    // Rows above the block must be untouched.
    EXPECT_TRUE(H.topRows(offset).isZero());

    // Every block ComputeJacobian wrote must be copied to the same columns.
    EXPECT_TRUE(block_matches(Index::Wsb));
    EXPECT_TRUE(block_matches(Index::Tsb));
    EXPECT_TRUE(block_matches(Index::Wbc));
    EXPECT_TRUE(block_matches(Index::Tbc));
    EXPECT_TRUE(block_matches(foff));

    // The reference-group pose: rotation at goff, translation at goff + 3.
    // The second of these is what the bug got wrong.
    EXPECT_TRUE(block_matches(goff));
    EXPECT_TRUE(block_matches(goff + 3));

    // Guard against the two group blocks being equal (or zero) for a degenerate
    // reason, which would let the old bug pass the assertions above.
    EXPECT_FALSE(f->J_.block(0, goff, 2, 3).isApprox(
        f->J_.block(0, goff + 3, 2, 3)));
    EXPECT_FALSE(f->J_.block(0, goff + 3, 2, 3).isZero());

    // The whole row block must equal J_: this also catches a block written to
    // the wrong offset that the named checks above do not cover.
    EXPECT_TRUE(H.middleRows(offset, 2).isApprox(f->J_));
}
