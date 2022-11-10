package com.example.kt.service
import okhttp3.MultipartBody
import okhttp3.ResponseBody
import retrofit2.Response
import retrofit2.http.*

interface FileAPI {
    @Multipart
    @POST("/api/smart_kc_test_uploader")
    suspend fun uploadFile(
        @Part file: MultipartBody.Part
    ): Response<ResponseBody>
}