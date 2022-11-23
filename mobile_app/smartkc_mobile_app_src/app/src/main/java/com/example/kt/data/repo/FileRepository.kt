package com.example.kt.data.repo

import android.net.Uri
import androidx.core.net.toFile
import com.example.kt.data.local.daos.FileDao
import com.example.kt.data.models.KTFile
import com.example.kt.service.FileAPI
import kotlinx.coroutines.flow.flow
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import javax.inject.Inject

class FileRepository @Inject constructor(
    private val fileDao: FileDao,
    private val fileAPI: FileAPI
){
    // insert new file record
    suspend fun insertNewFileRecord(uri: String, fileName: String) { fileDao.insertFileRecord(uri, fileName) }

    // get un-uploaded file records
    suspend fun getUnUploadedFileRecords(): List<KTFile> { return fileDao.getUnUploadedFiles() }

    // update uploaded file
    suspend fun markFileUploaded(uri: String) { fileDao.markFileUploaded(uri) }

    // upload file
    suspend fun uploadFile(ktFile: KTFile) = flow {
        val fileUri = Uri.parse(ktFile.uri)
        // Get file
        val file = fileUri.toFile()
        val requestFile = file.asRequestBody("image/*".toMediaTypeOrNull())
        val filePart = MultipartBody.Part.createFormData("file", ktFile.fileName, requestFile)
        val response = fileAPI.uploadFile(filePart)
        val responseBody = response.body()

        if (!response.isSuccessful) {
            error("Upload file API request failed")
        }

        if (responseBody != null) {
            markFileUploaded(ktFile.uri)
            emit(responseBody)
        } else {
            error("Request failed, response body was null")
        }
    }
}